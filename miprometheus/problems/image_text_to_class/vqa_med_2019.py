import os
import tqdm
import nltk
import pandas as pd
import torch
import pickle
from PIL import Image
from torchvision import transforms

from miprometheus.utils.problems_utils.language import Language
from miprometheus.utils.param_interface import ParamInterface
from miprometheus.problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem


class VQAMED(ImageTextToClassProblem):

    def __init__(self, params):
        super(VQAMED, self).__init__(params)
        self.name = 'VQAMED'
        self.embedding_source = 'VQAMED'
        self.parse_param_tree(params)
        self.data, self.word_dic, self.answer_dic = self.load_questions()

        # --> At this point, self.data contains the processed questions
        self.length = len(self.data)

        # check if the folder /generated_files in self.data_folder already exists, if not create it:
        if not os.path.isdir(os.path.join(self.data_folder, 'generated_files')):
            self.logger.warning('Folder {} not found, creating it.'.format(os.path.join(self.data_folder,
                                                                                        'generated_files')))
            os.mkdir(os.path.join(self.data_folder, 'generated_files'))

        # create the objects for the specified embeddings
        if self.embedding_type == 'random':
            self.logger.info('Constructing random embeddings using a uniform distribution')
            # instantiate nn.Embeddings look-up-table with specified embedding_dim
            self.n_vocab = len(self.word_dic)+1
            self.embed_layer = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.embedding_dim)

            # we have to make sure that the weights are the same during training and validation
            weights_filepath = os.path.join(self.data_folder, 'generated_files', '{}_embedding_weights.pkl'.format(self.embedding_source))
            if os.path.isfile(weights_filepath):
                self.logger.info('Found random embedding weights on file ({}), using them.'.format(weights_filepath))
                with open(weights_filepath, 'rb') as f:
                    self.embed_layer.weight.data = pickle.load(f)
            else:
                self.logger.warning('No weights found on file for random embeddings. Initializing them from a Uniform '
                                    'distribution and saving to file in {}'.format(weights_filepath))
                self.embed_layer.weight.data.uniform_(0, 1)
                with open(weights_filepath, 'wb') as f:
                    pickle.dump(self.embed_layer.weight.data, f)

        else:
            self.logger.info('Constructing embeddings using {}'.format(self.embedding_type))
            # instantiate Language class
            self.language = Language('lang')
            self.questions = [q['string_question'] for q in self.data]
            # use the questions set to construct the embeddings vectors
            self.language.build_pretrained_vocab(self.questions, vectors=self.embedding_type)

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {
            "height": self.height,
            "width": self.width,
            "depth": 3,
            'num_classes': self.nb_classes,
            "question_encoding_size": self.embedding_dim
            }

        # Define the data_definitions dict: holds a description of the DataDict content.
        self.data_definitions = {
            'images': {'size': [-1, 3, self.height, self.width], 'type': [torch.Tensor]},
            'questions': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
            'questions_length': {'size': [-1 ,1], 'type': [list, int]},
            'questions_string': {'size': [-1, -1], 'type': [list, str]},
            'targets': {'size': [-1, -1], 'type': [list, int]},

            'targets_string': {'size': [-1, 1], 'type': [list, str]},
            'index': {'size': [-1], 'type': [list, int]},
            'image_id': {'size': [-1, -1], 'type': [list, str]}
            }


    def parse_param_tree(self, params):
        # Load question settings.
        self.embedding_type = params['question']['embedding_type']
        self.embedding_dim = params['question']['embedding_dim']

        # Load image size.
        self.height = params['image']['height']
        self.width = params['image']['width']
        self.num_channels = 3

        # Retrieve path and expand it.
        self.data_folder = os.path.expanduser(params['settings']['data_folder'])
        # Set split-dependent data.
        if params['settings']['split'] == 'train':
            self.split = 'train'
            self.image_source = os.path.join(self.data_folder, 'Train_images')
        elif params['settings']['split'] == 'valid':
            self.split = 'valid'
            self.image_source = os.path.join(self.data_folder, 'Val_images')


    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: DataDict()
        """
        # Get item.
        item = self.data[index]

        # Load adequate image.
        img_id = item["image_id"]
        extension = '.jpg'
        with open(os.path.join(self.image_source, img_id + extension),'rb') as f:
            # Load image.
            img = Image.open(f).convert('RGB')
            # Resize it and transform to Torch Tensor.
            transfroms_com = transforms.Compose([
                    transforms.Resize([self.height,self.width]),
                        transforms.ToTensor()
                    ])
            img = transfroms_com(img).type(torch.FloatTensor).squeeze()

        # Process question.
        question = item["tokenized_question"]
        if self.embedding_type == 'random':
            # embed question:
            question = self.embed_layer(torch.LongTensor(question)).type(torch.FloatTensor)
        else:
            # embed question
            question = self.language.embed_sentence(item["string_question"])
        # Get length.
        question_length = question.shape[0]

        # Create the resulting data dict.
        data_dict = self.create_data_dict()

        data_dict['images'] = img
        data_dict['questions'] = question
        data_dict['questions_length'] = question_length
        data_dict['questions_string'] = item["string_question"]
        data_dict['targets'] = item["answer_encoded"]
        data_dict['target_string'] = item["string_answer"]
        data_dict['index'] = index
        data_dict['image_id'] = img_id

        return data_dict


    def collate_fn(self, batch):
        """
        Combines a list of DataDict (retrieved with :py:func:`__getitem__`) into a batch.
        Additionally pads all questions.

        .. warning:
            For now I am taking element 0 from the encoded target, so can apply our VQA models!

        :param batch: list of individual samples to combine
        :type batch: list

        :return: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', 'targets', \
        'targets_string', 'index','imgfiles'})

        """
        # Get batch size.
        batch_size = len(batch)

        # Create tensor of shape [batch_size x max question_length x embedding].
        max_len = max(map(lambda x: x['questions_length'], batch))
        questions = torch.zeros(batch_size, max_len, self.embedding_dim).type(torch.FloatTensor)
        for i, length in enumerate([item['questions_length'] for item in batch]): 
            questions[i, :length, :] = batch[i]['questions']

        # construct the DataDict and fill it with the batch
        data_dict = self.create_data_dict()

        data_dict['images'] = torch.stack([item['images'] for item in batch]).type(torch.FloatTensor)
        data_dict['questions'] = questions

        data_dict['questions_length'] = [item['questions_length'] for item in batch]
        data_dict['questions_string'] = [item['questions_string'] for item in batch]

        # Answer is a single item.
        #data_dict['targets'] = [item['targets'] for item in batch] 
        data_dict['targets'] = torch.tensor([item['targets'] for item in batch])

        data_dict['target_string'] =  [item['target_string'] for item in batch]
        data_dict['index'] = [item['index'] for item in batch]
        data_dict['image_id'] = [item['image_id'] for item in batch]

        return data_dict
        

    def load_questions(self):
        if self.split == 'train':
            question_file = os.path.join(self.data_folder, 'All_QA_Pairs_train.txt')
        elif self.split == 'valid':
            question_file = os.path.join(self.data_folder, 'All_QA_Pairs_val.txt')

        self.logger.info('Loading questions from {} ...'.format(question_file))
        print(question_file)
        df = pd.read_csv(filepath_or_buffer=question_file, sep='|',header=None,
                         names=['ImageID','Question','Answer'])

        result = []
        # Question related variables.
        question_dict = {}
        question_index = 1  # 0 reserved for padding
        # Answer related variables.
        answer_dict = {}
        answer_index = 0 # No padding here.

        t = tqdm.tqdm(total=len(df.index))

        for index, row in df.iterrows():
            image_id = row['ImageID']
            question = row['Question']
            question_words = nltk.word_tokenize(question)
            question_token = []

            for qword in question_words:
                try:
                    question_token.append(question_dict[qword])
                except KeyError:
                    question_token.append(question_index)
                    question_dict[qword] = question_index
                    question_index += 1

            # Process answer.
            # We assume this is a classification problem, so need to create a seperate class for each answer.
            # Number of classes = number of possible answers.
            answer = row['Answer']

            try:
                answer_encoded = answer_dict[answer]
            except KeyError:
                # New word.
                answer_dict[answer] = answer_index
                answer_encoded = answer_index
                answer_index += 1
                
            # Add record to result.
            result.append({
                'tokenized_question': question_token,
                'string_question': question,
                'answer_encoded': answer_encoded,
                'string_answer': answer,
                'image_id': image_id})

            t.update()
        t.close()

        # Set number of answer classes.
        self.nb_classes = len(answer_dict)
        self.logger.info('Constructed question word dictionary of length {}'.format(len(question_dict)))
        self.logger.info('Constructed answer word dictionary of length {}'.format(len(answer_dict)))

        return result, question_dict, answer_dict


if __name__ == '__main__':
    params = ParamInterface()
    params.add_config_params({'settings': {'data_folder':'~/data/ImageClef-2019-VQA-Med-Training',
                                           'split': 'train',
                                           'dataset_variant': 'VQAMed2019'},
                              'images': {'raw_images': True,},
                              'questions': {'embedding_type': 'random', 'embedding_dim': 300}})
    vqa_med_train_dataset = VQAMED(params)
    sample = vqa_med_train_dataset[1]
    print(repr(sample))
    print('__getitem__ works.')
