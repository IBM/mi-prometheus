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
from miprometheus.utils.data_dict import DataDict
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

        # define the default_values dict: holds parameters values that a model may need.
        self.default_values = {
                'num_classes': self.nb_classes,
                "height": self.height,
                "width": self.width,
                "num_channels": 3,
                "question_size": 300
        }

    def parse_param_tree(self, params):
        self.data_folder = params['settings']['data_folder']
        self.embedding_type = params['questions']['embedding_type']
        self.embedding_dim = params['questions']['embedding_dim']

        self.question_encoding_size = self.embedding_dim
        self.height = 224
        self.width = 224
        self.num_channels = 3

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

        item = self.data[index]
        question = item["tokenized_question"]
        question_string = item["string_question"]
        answer = item["tokenized_answer"]
        answer_string = item["string_answer"]
        img_id = item["image_id"]

        extension = '.jpg'
        with open(os.path.join(self.image_source, img_id + extension),'rb') as f:
            try:
                img = torch.load(f)
                img = torch.from_numpy(img).type(torch.FloatTensor).squeeze()
            except Exception:
                img = Image.open(f).convert('RGB')  # for the original images
                # img = transforms.Resize(224,224)(img)
                transfroms_com = transforms.Compose([
                                                    transforms.Resize([128,128]),
                                                     transforms.ToTensor()
                                                    ])
                img = transfroms_com(img).type(torch.FloatTensor).squeeze()

                # img = transforms.ToTensor()(img).type(torch.FloatTensor).squeeze()

        if self.embedding_type == 'random':
            # embed question:
            question = self.embed_layer(torch.LongTensor(question)).type(torch.FloatTensor)
        else:
            # embed question
            question = self.language.embed_sentence(question_string)

        question_length = question.shape[0]

        data_dict = DataDict()
        data_dict['images'] = img
        data_dict['questions'] = question
        data_dict['questions_length'] = question_length
        data_dict['questions_string'] = question_string
        data_dict['targets'] = answer
        data_dict['target_string'] = answer_string
        data_dict['index'] = index
        data_dict['image_id'] = img_id

        return data_dict

    def collate_fn(self, batch):
        return self.__getitem__(0)

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
        question_dict = {}
        answer_dict = {}
        question_index = 1  # 0 reserved for padding
        answer_index = 1  # 0 reserved for padding

        t = tqdm.tqdm(total=len(df.index))
        num_classes = set()
        for index, row in df.iterrows():
            image_id = row['ImageID']
            question = row['Question']
            question_words = nltk.word_tokenize(question)
            question_token = []

            for qword in question_words:
                try:
                    question_token.append(question_dict[qword])
                except Exception:
                    question_token.append(question_index)
                    question_dict[qword] = question_index
                    question_index += 1

            answer = row['Answer']
            num_classes.add(answer)
            answer_words = nltk.word_tokenize(answer)
            answer_token = []

            for aword in answer_words:
                try:
                    answer_token.append(answer_dict[aword])
                except:
                    answer_token.append(answer_index)
                    answer_dict[aword] = answer_index
                    answer_index += 1

            result.append({'tokenized_question': question_token,
                          'string_question': question,
                           'tokenized_answer': answer_token,
                          'string_answer': answer,
                          'image_id': image_id})

            t.update()
        t.close()

        self.nb_classes = len(num_classes)
        self.logger.info('Constructed question word dictionary of length {}'.format(len(question_dict)))
        self.logger.info('Constructed answer word dictionary of length {}'.format(len(answer_dict)))

        return result, question_dict, answer_dict


if __name__ == '__main__':
    params = ParamInterface()
    params.add_config_params({'settings': {'data_folder':'/raid/data/cshivade/project_resources/radvisdial/imageclef_vqa_2019/ImageClef-2019-VQA-Med-Training',
                                           'split': 'train',
                                           'dataset_variant': 'VQAMed2019'},
                              'images': {'raw_images': True,},
                              'questions': {'embedding_type': 'random', 'embedding_dim': 300}})
    vqa_med_train_dataset = VQAMED(params)
    sample = vqa_med_train_dataset[1]
    print(repr(sample))
    print('__getitem__ works.')
