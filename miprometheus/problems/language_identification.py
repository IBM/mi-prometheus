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

#torch.manual_seed(1)

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
        self.data_definitions = {'encoded_inputs': {'size': [-1, -1], 'type': [torch.Tensor]}, # encoded with BoW its is [BATCH_SIZE x VOCAB_SIZE] !
                                 'sententes': {'size': [-1, 1], 'type': [list, str]}, # [BATCH x SENTENCE (list of words as single string)]
                                 'target_classes': {'size': [-1, -1], 'type': [torch.Tensor]},
                                 'target_labels': {'size': [-1, 1], 'type': [list, str]}
                                 }

        # Dummy data.
        self.data = [("me gusta comer en la cafeteria", "SPANISH"),
                ("Give it to me", "ENGLISH"),
                ("No creo que sea una buena idea", "SPANISH"),
                ("No it is not a good idea to get lost at sea", "ENGLISH")]

        self.test_data = [("Yo creo que si", "SPANISH"),
                    ("it is lost on me", "ENGLISH")]

        # word_to_ix maps each word in the vocab to a unique integer, which will be its
        # index into the Bag of words vector
        self.word_to_ix = {}
        for sent, _ in self.data + self.test_data:
            for word in sent.split():
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
        #print(word_to_ix)

        self.label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

        # Set length.
        self.length = len(self.data)

        self.num_classes = len(self.label_to_ix)
        self.item_size = len(self.word_to_ix)

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'num_classes': self.num_classes,
                               'input_size': self.item_size
                               }

        # Set loss.
        self.loss_function = nn.NLLLoss()


    def make_bow_vector(self, sentence):
        vec = torch.zeros(len(self.word_to_ix))
        for word in sentence.split():
            vec[self.word_to_ix[word]] += 1
        return vec


    def make_target(self, label):
        return torch.LongTensor([self.label_to_ix[label]])

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
        data_dict['encoded_inputs'] = self.make_bow_vector(sentence)
        data_dict['sententes'] = sentence
        data_dict['target_classes'] = self.make_target(language)
        data_dict['target_labels'] = language
        return data_dict

    def evaluate_loss(self, data_dict, logits):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.

        :param data_dict: DataDict containing "target_classes" field.
        :param logits: Predictions (log_probs) being output of the model.

        """
        targets = data_dict['target_classes'].squeeze(dim=1)
        loss = self.loss_function(logits, targets)
        return loss


#class BOWEncoder(object):
#    def  __init__(self):
#        self = word_to_ix
#
#    def encode():
#        pass



class BoWClassifier(nn.Module): 

    def __init__(self, default_values):
        # Call parent constructor.
        super(BoWClassifier, self).__init__()

        # Retrieve input (vocabulary) size and number of classes from default params.
        self.input_size = default_values['input_size']
        self.num_classes = default_values['num_classes']

        # Simple classifier.
        self.linear = nn.Linear(self.input_size, self.num_classes)


    def forward(self, data_dict):
        """
        forward pass of the  model.

        :param data_dict: DataDict({'encoded_inputs', ...}), where:

            - encoded_input: [batch_size, input_size],

        :return: Predictions (log_probs) [batch_size, num_classes]

        """
        # get images
        inputs = data_dict['encoded_inputs']
        return F.log_softmax(self.linear(inputs), dim=1)


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
    model = BoWClassifier(problem.default_values)

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
    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()


            # Step 3. Run our forward pass.
            log_probs = model(batch)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = problem.evaluate_loss(batch, log_probs)

            loss.backward()
            optimizer.step()

    #with torch.no_grad():
    #    for instance, label in test_data:
    #        bow_vec = make_bow_vector(instance, word_to_ix)
    #        log_probs = model(bow_vec)
    #        #print(log_probs)

    # Index corresponding to Spanish goes up, English goes down!
    #print(next(model.parameters())[:, word_to_ix["creo"]])