# Thoma, Martin. "The WiLI benchmark dataset for written language identification." arXiv preprint arXiv:1801.07779 (2018).
# https://arxiv.org/abs/1801.07779
# https://zenodo.org/record/841984/files/wili-2018.zip?download=1


# Author: Robert Guthrie

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from miprometheus.problems import Problem
from miprometheus.utils.data_dict import DataDict

#torch.manual_seed(1)
class Component(object):
    def __init__(self):
        # Set default (empty) data definitions and default_values.
        self.data_definitions = {}
        self.default_values =  {}

    def create_data_dict(self, data_definitions = None):
        """
        Returns a :py:class:`miprometheus.utils.DataDict` object with keys created on the \
        problem data_definitions and empty values (None).

        :param data_definitions: Data definitions that will be used (DEFAULT: None, meaninng that self.data_definitions will be used)

        :return: new :py:class:`miprometheus.utils.DataDict` object.
        """
        # Use self.data_definitions as default.
        data_definitions = data_definitions if data_definitions is not None else self.data_definitions

        return DataDict({key: None for key in data_definitions.keys()})

    def extend_data_dict(self, data_dict, data_definitions): #= None):
        """
        Copies and optionally extends a :py:class:`miprometheus.utils.DataDict` object by adding keys created on the \
        problem data_definitions and empty values (None).

        .. warning::
            This is in-place operation, i.e. extends existing object, does not return a new one.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object.

        :param data_definitions: Data definitions that will be used (DEFAULT: None, meaninng that self.data_definitions will be used)

        """
        # Use self.data_definitions as default.
        #data_definitions = data_definitions if data_definitions is not None else self.data_definitions
        for key in data_definitions.keys():
            data_dict[key] = None

    def extend_default_values(self, default_values):
        """
        Extends the input list of default values by component's own list.

        .. warning::
            This is in-place operation, i.e. extends existing object, does not return a new one.

        .. warning::
            Value will be overwritten for the existing keys!

        :param default_values: Dictionary containing default values (that will be extended).
        """
        for key, value in self.default_values.items():
            default_values[key] = value


class SoftmaxClassifier(nn.Module, Component): 
    """
    Simple Classifier consisting of fully connected layer with log softmax non-linearity.
    """
    def __init__(self, params, input_default_values):
        """
        Initializes the classifier.

        :param params_: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param default_input_values_: Dictionary containing default_input_values.
        """
        # Call parent constructor.
        super(SoftmaxClassifier, self).__init__()
        self.name = "SoftmaxClassifier"

        # Retrieve input (vocabulary) size and number of classes from default params.
        self.input_size = input_default_values['encoded_input_size']
        self.num_classes = input_default_values['num_classes']

        # Simple classifier.
        self.linear = nn.Linear(self.input_size, self.num_classes)
        
        # Set default data_definitions dict.
        # Encoded with BoW its is [BATCH_SIZE x NUM_CLASSES] !
        self.data_definitions = {'predictions': {'size': [-1, self.num_classes], 'type': [torch.Tensor]} }


    def forward(self, data_dict):
        """
        Forward pass of the  model.

        :param data_dict: DataDict({'encoded_inputs', ...}), where:

            - encoded_inputs: [batch_size, input_size],

        :return: Predictions (log_probs) [batch_size, target_size]

        """
        inputs = data_dict['encoded_inputs']
        predictions = F.log_softmax(self.linear(inputs), dim=1)
        # Add them to datadict.
        self.extend_data_dict(data_dict, {'predictions': None})
        data_dict["predictions"] = predictions
        # For compatibility. TODO: remove return
        return data_dict

    
class LanguageIdentification(Problem):
    """
    Language identification (classification) problem.
    """

    def __init__(self, params):
        """
        Initializes problem object. Calls base constructor.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        super(LanguageIdentification, self).__init__(params)
        self.name = 'LanguageIdentification'

        # Set default parameters.
        self.params.add_default_params({'data_folder': '~/data/language_identification',
                                        'use_train_data': True
                                        })
        # Get absolute path.
        #data_folder = os.path.expanduser(self.params['data_folder'])
        # Retrieve parameters from the dictionary.
        self.use_train_data = self.params['use_train_data']


        # Set default data_definitions dict.
        self.data_definitions = {'sentences': {'size': [-1, 1], 'type': [list, str]}, 
                                # [BATCH x SENTENCE (list of words as a single string)]
                                'languages': {'size': [-1, 1], 'type': [list, str]}
                                # [BATCH x WORD (word as a single string)]
                                }

        # Dummy data.
        self.data = [("me gusta comer en la cafeteria", "SPANISH"),
                ("Give it to me", "ENGLISH"),
                ("No creo que sea una buena idea", "SPANISH"),
                ("No it is not a good idea to get lost at sea", "ENGLISH")]

        self.test_data = [("Yo creo que si", "SPANISH"),
                    ("it is lost on me", "ENGLISH")]

        # Set length.
        self.length = len(self.data)

        # Set loss.
        self.loss_function = nn.NLLLoss()


    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'images','targets', 'targets_label'})``, with:

            - images: Image, resized if ``self.resize`` is set,
        """
        # Get sentence and language.
        (sentence, language) = self.data[index]

        # Return data_dict.
        data_dict = self.create_data_dict()
        data_dict['sentences'] = sentence
        data_dict['languages'] = language
        return data_dict

    def evaluate_loss(self, data_dict):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.

        :param data_dict: DataDict containing:
            - "encoded_targets": batch of targets (class indices) [BATCH_SIZE x NUM_CLASSES]

            - "predictions": batch of predictions (log_probs) being outputs of the model [BATCH_SIZE x 1]

        :return: loss calculated useing the loss function (negative log-likelihood).
        """
        targets = data_dict['encoded_targets']
        predictions = data_dict['predictions']
        loss = self.loss_function(predictions, targets.squeeze(dim=1))
        return loss


class Encoder(Component):
    """
    Default encoder class. Creates interface and provides generic methods for batch processing.
    """
    def __init__(self, name_, params_, default_input_values_):
        """
        Initializes encoder object.

        :param name_: Name of the encoder.

        :param params_: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param default_input_values_: Dictionary containing default_input_values.
        """
        # Save name and params.
        self.name = name_
        self.params = params_

    def encode_sample(self, sample):
        """
        Method responsible for encoding of a single sample (interface).
        """
        pass

    def encode_batch(self, data_dict):
        """
        Method responsible for encoding of a single sample (interface).

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing data to encode and that will be extended with encoded results.
        """
        pass

    def decode_sample(self, encoded_sample):
        """
        Method responsible for decoding of a single encoded sample (interface).
        """
        pass    

    def decode_batch(self, data_dict):
        """
        Method responsible for decoding of a single encoded sample (interface).

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing data to decode and that will be extended with decoded results.
        """
        pass    


class BOWSentenceEncoder(Encoder):
    """
    Simple Bag-of-word type encoder that encodes the sentence into a vector.
    
    .. warning::
        BoW transformation is inreversible, thus decode-related methods in fact return original inputs.
    """
    def  __init__(self, params_, default_input_values_):
        """t
        Initializes the bag-of-word encoded by creating dictionary mapping ALL words from training, validation and test sets into unique indices.

        :param name_: Name of the encoder.

        :param params_: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param default_input_values_: Dictionary containing default_input_values.
        """
        # Call base class constructor.
        super(BOWSentenceEncoder, self).__init__('BOWSentenceEncoder', params_, default_input_values_)
        
        # Dummy data.
        self.data = [("me gusta comer en la cafeteria", "SPANISH"),
                ("Give it to me", "ENGLISH"),
                ("No creo que sea una buena idea", "SPANISH"),
                ("No it is not a good idea to get lost at sea", "ENGLISH")]

        self.test_data = [("Yo creo que si", "SPANISH"),
                    ("it is lost on me", "ENGLISH")]

        # Dictionary word_to_ix maps each word in the vocab to a unique integer.
        # It will later used as word index during encoding into the Bag of words vector.
        self.word_to_ix = {}
        for sent, _ in self.data + self.test_data:
            for word in sent.split():
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
        #print(word_to_ix)

        # Size of a single encoded item.
        self.item_size = len(self.word_to_ix)

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'encoded_input_size': self.item_size}
        # Set default data_definitions dict.
        # Encoded with BoW its is [BATCH_SIZE x VOCAB_SIZE] !
        self.data_definitions = {'encoded_inputs': {'size': [-1, -1], 'type': [torch.Tensor]} }

    def encode_batch(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of batch ("sencentes").
        Stores result in "encoded_inputs" field of data_dict.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing (among others):

            - "sencentes": expected input containing list of samples [BATCH SIZE] x [string]
            - "encoded_inputs": added output tensor with encoded samples [BATCH_SIZE x INPUT_SIZE]
        """
        # Get inputs to be encoded.
        batch = data_dict["sentences"]
        encoded_batch_list = []
        # Process samples 1 by one.
        for sample in batch:
            # Encode sample
            encoded_sample = self.encode_sample(sample)
            # Add to list plus unsqueeze batch dimension(!)
            encoded_batch_list.append( encoded_sample.unsqueeze(0) )
        # Concatenate batch.
        encoded_batch = torch.cat(encoded_batch_list, dim=0)
        # Create the returned dict.
        self.extend_data_dict(data_dict, {'encoded_inputs': None})
        data_dict["encoded_inputs"] = encoded_batch


    def encode_sample(self, sentence):
        """
        Generates a bag-of-word vector of length `encoded_input_size`.

        :return: torch.LongTensor [INPUT_SIZE]
        """
        # Create empty vector.
        vector = torch.zeros(len(self.word_to_ix))
        # Encode each word and add its "representation" to vector.
        for word in sentence.split():
            vector[self.word_to_ix[word]] += 1
        return vector


    def decode_batch(self, data_dict):
        """ 
        Method DOES NOT change data dict.
        """ 
        pass

    def decode_sample(self, vector):
        """
        Method DOES NOT anything, as bow transformation cannot be reverted.
        """
        pass

class WordEncoder(Encoder):
    """
    Simple word encoder. Encodes a given input word into a unique index.
    """
    def  __init__(self, params_, default_input_values_):
        """
        Initializes the simple word encoded by creating dictionary mapping ALL words from training, validation and test sets into unique indices.

        :param name_: Name of the encoder.

        :param params_: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param default_input_values_: Dictionary containing default_input_values.
        """
        # Call base class constructor.
        super(WordEncoder, self).__init__('WordEncoder', params_, default_input_values_)

        # Dummy data.
        self.word_to_ix = {"SPANISH": 0, "ENGLISH": 1}
        self.ix_to_word = ["SPANISH", "ENGLISH"]
        
        self.num_classes = len(self.word_to_ix)

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'num_classes': self.num_classes}
        # Set default data_definitions dict.
        self.data_definitions = {
            'encoded_targets': {'size': [-1, -1], 'type': [torch.Tensor]},
            'decoded_predictions': {'size': [-1, 1], 'type': [list, str]}
            }

    def encode_batch(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of bach ("targets").
        Stores result in "encoded_inputs" field of in data_dict.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing (among others):

            - "targets": expected input field containing list of words

            - "encoded_targets": added field containing output, tensor with encoded samples [BATCH_SIZE x 1] 
        """
        # Get inputs to be encoded.
        batch = data_dict["languages"]
        encoded_targets_list = []
        # Process samples 1 by one.
        for sample in batch:
            # Encode sample
            encoded_sample = self.encode_sample(sample)
            # Add to list plus unsqueeze batch dimension(!)
            encoded_targets_list.append( encoded_sample.unsqueeze(0) )
        # Concatenate batch.
        encoded_targets = torch.cat(encoded_targets_list, dim=0)
        # Create the returned dict.
        self.extend_data_dict(data_dict, {'encoded_targets': None})
        data_dict["encoded_targets"] = encoded_targets

    def encode_sample(self, word):
        """
        Encodes a single word.

        :param word: A single word (string).

        :return: torch.LongTensor [1] (i.e. tensor of size 1)
        """
        return torch.LongTensor([self.word_to_ix[word]])

    def decode_batch(self, data_dict):
        """ 
        Method adds decoded predictions to data dict.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing (among others):

            - "predictions": expected input field containing predictions, tensor [BATCH_SIZE x NUM_CLASSES]
            - "decoded_predictions": returned list of words [BATCH_SIZE] x [string] 
        """ 
        # Get inputs to be encoded.
        predictions = data_dict["predictions"]
        decoded_predictions_list = []
        # Process samples 1 by one.
        for sample in predictions.chunk(predictions.size(0), 0):
            # Decode sample
            decoded_sample = self.decode_sample(sample)
            # Add to list plus unsqueeze batch dimension(!)
            decoded_predictions_list.append( decoded_sample )

        # Create the returned dict.
        self.extend_data_dict(data_dict, {'decoded_predictions': None})
        data_dict["decoded_predictions"] = decoded_predictions_list


    def decode_sample(self, vector):
        """
        Decodes vector into a single word.
        Handles with two types of inputs:

            - a single index: returns the associated word.
         
            - a vector containing a probability distribution: returns word associated with index with with max probability.

        :param vector: Single index or vector containing a probability distribution.

        :return: torch.LongTensor [1] (i.e. tensor of size 1)
        """
        return self.ix_to_word[vector.argmax(dim=1)]


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # Load parameters.
    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()  # using the default values

    # Test different options.
    params.add_config_params({'data_folder': '~/data/mnist',
                                    'use_train_data': True
                                    })
    batch_size = 2

    # Create problem and model.
    problem  = LanguageIdentification(params)
    default_values = problem.default_values
    # Input (sentence) encoder.
    input_encoder = BOWSentenceEncoder(params, default_values)
    input_encoder.extend_default_values(default_values)
    # Output (word) encoder.
    output_encoder = WordEncoder(params, default_values)
    output_encoder.extend_default_values(default_values)
    # Model.
    model = SoftmaxClassifier(params, default_values)

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=problem, collate_fn=problem.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    # Print the matrix column corresponding to "creo"
    #print(next(model.parameters())[:, word_to_ix["creo"]])

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(1000):
        for i, batch in enumerate(dataloader):
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Encode inputs and targets.
            input_encoder.encode_batch(batch)
            output_encoder.encode_batch(batch)

            # Step 3. Run our forward pass.
            model(batch)
            
            # Decode predictions.
            output_encoder.decode_batch(batch)
            print("sequence: {} targets: {} model predictions: {}".format(batch["sentences"], batch["languages"], batch["decoded_predictions"]))

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = problem.evaluate_loss(batch)
            #print("Loss = ", loss)

            loss.backward()
            optimizer.step()
