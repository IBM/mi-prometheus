#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sort_of_clevr.py: ``Sort-of-CLEVR`` is a simplified version of the ``CLEVR`` dataset.

"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import h5py
import numpy as np
from PIL import Image, ImageDraw
import progressbar
import os

import torch
from problems.problem import DataDict
from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem, ObjectRepresentation


class SortOfCLEVR(ImageTextToClassProblem):
    """
    ``Sort-of-CLEVR`` is a simple VQA problem, where the goal is to answer the\
    question regarding a given image. Implementation of the generation is\
    inspired by: git@github.com:gitlimlab/Relation-Network-Tensorflow.git

    Improvements:

        - Generates scenes with dynamic varying number of objects (2-6)
        - More types of intra- and inter-relational questions
        - More natural interpretation of questions

    :param data_folder: folder where to look for or save the file containing the dataset
    :type data_folder: str

    :param split: Indicates either ``train``, ``test`` or ``val``
    :type split: str

    :param img_size: Size of the images to generate.
    :type img_size: int

    :param dataset_size: How many samples to generate.
    :type dataset_size: int

    :param regenerate: Whether to regenerate the dataset
    :type regenerate: Bool

    .. note::

        When generating the dataset, this class:

            - First verifies if a file with a matching filename already exists in the ``data_folder``.
              The filename follows the following template:

                >>> filename = '<split>_<dataset_size>_<img_size>.hy'


            - If such a file exists, it is loaded and used as the dataset. If not, it is created and then used.
            - If ``regenerate`` is ``True``, the file is recreated regardless if one with the matching filename\
              already exists or not.


    """

    def __init__(self, params):
        """
        Initializes ``Sort-of-CLEVR`` problem, calls base class ``ImageTextToClassProblem``\
         initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """

        # Call base class constructors.
        super(SortOfCLEVR, self).__init__(params)

        # problem name
        self.name = 'Sort-of-CLEVR'

        # parse params
        self.img_size = params["img_size"]
        self.dataset_size = params["dataset_size"]
        self.regenerate = params.get("regenerate", False)

        # Set general color properties.
        self.BG_COLOR = (180, 180, 150)
        self.COLOR = [
            (0, 0, 210),    # 'blue'
            (0, 210, 0),    # 'green'
            (210, 0, 0),    # 'red'
            (150, 150, 0),  # 'yellow'
            (150, 0, 150),  # 'magenta'
            (0, 150, 150),  # 'cyan'
            # add more colors here if needed
        ]

        # Other hardcoded parameters.
        self.NUM_SHAPES = 2
        self.NUM_COLORS = len(self.COLOR)
        self.NUM_QUESTIONS = 7

        # Objects are characterised by colors, so cannot have more objects than
        # colors.
        self.MAX_NUM_OBJECTS = min(6, self.NUM_COLORS)
        self.GRID_SIZE = 4

        # Get absolute path.
        data_folder = os.path.expanduser(params['data_folder'])

        # create the folder if it doesn't exist
        if not os.path.isdir(data_folder):
            self.logger.warning('Indicated data_folder does not exist, creating it.')
            os.mkdir(data_folder)

        # construct the dataset filename from 3 values:
        # set: either 'train', 'test' or 'val'
        # dataset size
        # image size
        data_filename = '{}_{}_{}.hy'.format(params['split'], str(self.dataset_size), str(self.img_size))

        # define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'img_size': self.img_size}

        # define the data_definitions dict: holds a description of the DataDict content
        self.data_definitions = {'images': {'size': [-1, 3, self.img_size, self.img_size], 'type': [torch.Tensor]},
                                 'questions': {'size': [-1, self.NUM_COLORS+self.NUM_QUESTIONS], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, self.NUM_COLORS+self.NUM_SHAPES+2], 'type': [torch.Tensor]},
                                 'targets_index': {'size': [-1], 'type': [torch.Tensor]},
                                 'scenes_description': {'size': [-1, -1], 'type': [list, str]},
                                 }

        # Load or generate the dataset.
        self.load_dataset(data_folder, data_filename)

        self.length = self.dataset_size

    def load_dataset(self, data_folder, data_filename):
        """
        Loads the dataset from the HDF5-encoded file.

        .. note::

            This function will look first if a dataset with the same filename already exists or not in\
             the specified ``data_folder`` (this filename contains the number of samples and image size of the\
              samples). If no such file does not exist, it is generated and saved in ``data_folder`` (with\
               the specified ``data_filename``).

        """
        # name of the file to look for or create
        self.filename = os.path.join(data_folder, data_filename)

        if self.regenerate:
            self.logger.warning('Regenerate is set to true: regenerating the dataset from scratch, '
                                'without looking for an existing one.')
            self.generate_h5py_dataset(self.filename)

        else:  # regenerate is false, looking if the file already exists
            if os.path.isfile(self.filename):
                self.logger.warning('Found file {}, using it as the dataset as it matches the filename template.'.format(self.filename))

            else:  # the file doesn't exist, we need to create it.
                self.logger.warning('File {} not found on disk, generating a new dataset.'.format(self.filename))
                self.generate_h5py_dataset(self.filename)

    def generate_h5py_dataset(self, filename):
        """
        Generates a whole new ``Sort-of-CLEVR`` dataset and saves it in the form of\
        a HDF5 file.

        :param filename: name of the file containing the samples.
        :type filename: str

        """
        # open the HDF5 file.
        file = h5py.File(filename, 'w')
        # progress bar
        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        count = 0

        while count < self.dataset_size:

            # Generate the scene.
            objects = self.generate_scene_representation()

            # Generate corresponding image, questions and answers.
            I = self.generate_image(objects)
            Q = self.generate_question_matrix(objects)
            A = self.generate_answer_matrix(objects)

            # Iterate through all questions generated for a given scene.
            for j in range(len(objects) * self.NUM_QUESTIONS):

                # Create new group.
                id = str(count)
                grp = file.create_group(id)

                # Set data.
                grp['image'] = I
                grp['question'] = Q[j, ...]
                grp['answer'] = A[j, ...]
                grp['scene_description'] = self.scene2str(objects)

                # Increment counter.
                count += 1

                # Update progress bar.
                if count % (self.dataset_size / 100) == 0:
                    bar.update(count / (self.dataset_size / 100))

                # Check whether we generated the required number of samples
                if count >= self.dataset_size:
                    break

        # Finalize the generation.
        bar.finish()
        file.close()
        self.logger.info('Generated dataset with {} samples and saved to {}'.format(self.dataset_size, self.filename))

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        .. warning::

            **HDF5 does not support multi threaded data access with num_workers > 1 on the data loading.**
            A way around this is to move every call for opening the HDF5 file to this ``__getitem__`` method.

            See https://discuss.pytorch.org/t/hdf5-multi-threaded-alternative/6189/9 for more info.

        :param index: index of the sample to return.

        :return: DataDict({'images','questions', 'targets', 'targets_index', 'scenes_description'}), with:

            - images: images (``self.img_size``)
            - questions: encoded questions
            - targets: one-hot encoded answers
            - targets_index: index of the answers
            - scenes_description: Scene description.

        """
        # load the file
        data = h5py.File(self.filename, 'r')
        sample = data[str(index)]

        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['images'] = (sample['image'].value / 255).transpose(2, 1, 0)
        data_dict['questions'] = sample['question'].value.astype(np.float32)
        data_dict['targets'] = sample['answer'].value.astype(np.float32)
        data_dict['targets_index'] = np.argmax(data_dict['targets'])
        data_dict['scenes_description'] = sample['scene_description'].value

        return data_dict

    def collate_fn(self, batch):
        """
        Combines a list of ``DataDict`` (retrieved with ``__getitem__`` ) into a batch.

        .. note::

            This function wraps a call to ``default_collate`` and simply returns the batch as a ``DataDict``\
            instead of a dict.

        :param batch: list of individual ``DataDict`` samples to combine.

        :return: ``DataDict({'images','questions', 'targets', 'targets_index', 'scenes_description'})`` containing the batch.

        """

        return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(SortOfCLEVR, self).collate_fn(batch).values())})

    def color2str(self, color_index):
        """
        Decodes the specified color index and returns it as a string.

        :param color_index: Index of the color.
        :type color_index: int

        :return: color name as a string.

        """
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
        }[color_index]

    def shape2str(self, shape_index):
        """
        Decodes the specified shape index and returns it as a string.

        :param shape_index: Index of the color.
        :type shape_index: int

        :return: shape name as a string.

        """
        return {
            0: 'rectangle',
            1: 'circle',
        }[shape_index]

    def question_type_template(self, question_index):
        """
        Decodes the specified question index and returns the corresponding string template.

        :param question_index: Index of the color.
        :type question_index: int

        :return: corresponding string template.

        """
        return {
            0: 'What is the shape of the {} object?',
            1: 'Is the {} {} closer to the bottom of the image?',
            2: 'Is the {} {} closer to the left side of the image?',
            3: 'What is the shape of the object nearest to the {} {}?',
            4: 'What is the shape of the object farthest from the {} {}?',
            5: 'What is the color of the object nearest to the {} {}?',
            6: 'What is the color of the object farthest from the {} {}?',
            # 7: 'How many objects have the same shape as the {} {}?,
        }[question_index]

    def question2str(self, encoded_question):
        """
        Decodes the encoded question, i.e. produces a human-understandable string.

        :param encoded_question: Concatenation of two one-hot vectors:

            - The first one denotes the object of interest (its color),
            - The second one denotes the question type.

        :type encoded_question: tensor

        :return: The question as a human-understandable string.

        """
        # "Decode" the color_query vector.
        color = np.argmax(encoded_question[:self.NUM_COLORS])
        question_code = np.argmax(encoded_question[self.NUM_COLORS:])

        # Return the question as a string.
        return self.question_type_template(question_code).format(self.color2str(color), 'object')

    def answer2str(self, encoded_answer):
        """
        Decodes the answer and returns the corresponding label.

        :param encoded_answer: Answer index, encoded as a one-hot vector.
        :type encoded_answer: np.array

        :return: answer label.

        """
        return {
            # 0-5 colors
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
            # 6-7 shapes
            6: 'rectangle',
            7: 'circle',
            # 8-9 yes/no
            8: 'yes',
            9: 'no',
        }[np.floor(encoded_answer)]

    def scene2str(self, objects):
        """
        Returns a string containing the shape, color and position of every object forming the scene.

        :param objects: List of objects - abstract scene representation.
        :type object: list

        :return: Str containing the scene description.

        """
        desc = '| '
        for obj in objects:
            # Add description
            desc = desc + ('{} {} at ({}, {}) | '.format(self.color2str(obj.color),
                                                         self.shape2str(obj.shape), obj.x, obj.y))
        return desc

    def generate_scene_representation(self):
        """
        Generates the scene representation.

        :return: List of objects - abstract scene representation.

        """
        # Generate list of objects - no more then the number of colors.
        num_objects = np.random.random_integers(2, self.MAX_NUM_OBJECTS)

        # Shuffle "grid positions".
        grid_positions = np.arange(self.GRID_SIZE * self.GRID_SIZE)
        np.random.shuffle(grid_positions)

        # Size of a "grid block".
        block_size = int(self.img_size * 0.9 / self.GRID_SIZE)

        # Shuffle colors.
        colors = np.arange(self.NUM_COLORS)
        np.random.shuffle(colors)
        colors = colors[:num_objects]

        # Generate shapes.
        shapes = (np.random.rand(num_objects) < 0.5).astype(int)

        # List of objects presents in the scene.
        objects = []

        # Generate coordinates.
        for i in range(num_objects):
            # Calculate object positions depending on "grid positions"
            x = grid_positions[i] % self.GRID_SIZE
            y = (
                self.GRID_SIZE -
                np.floor(
                    grid_positions[i] /
                    self.GRID_SIZE) -
                1).astype(
                np.uint8)
            # Calculate "image coordinates".
            x_img = (x + 0.5) * block_size + np.random.random_integers(-2, 2)
            y_img = (y + 0.5) * block_size + np.random.random_integers(-2, 2)
            # Add object to list.
            objects.append(ObjectRepresentation(
                x_img, y_img, colors[i], shapes[i]))

        return objects

    def generate_image(self, objects):
        """
        Generates the image on the basis of a given scene representation.

        :param objects: List of objects - abstract scene representation.
        :type object: list

        :return: ``np.array`` containing the generated image.

        """
        img_size = self.img_size
        shape_size = int((img_size * 0.9 / self.GRID_SIZE) * 0.7 / 2)

        # Generate image [img_size, img_size, 3]
        img = Image.new('RGB', (img_size, img_size), color=self.BG_COLOR)
        drawer = ImageDraw.Draw(img)

        for obj in objects:
            # Calculate object position.
            position = (obj.x - shape_size, obj.y - shape_size,
                        obj.x + shape_size, obj.y + shape_size)
            # Draw object.
            if obj.shape == 1:
                drawer.ellipse(position, fill=self.COLOR[obj.color])
            else:
                drawer.rectangle(position, fill=self.COLOR[obj.color])

        # Cast to np.
        return np.array(img)

    def generate_question_matrix(self, objects):
        """
        Generates the questions matrix: [# of shape * # of Q, # of color + # of Q].

        This matrix contains all possible questions for a given scene representation.


        :param objects: List of objects - abstract scene representation.
        :type object: list

        :return the questions matrix (``np.array``)
        """
        Q = np.zeros((len(objects) * self.NUM_QUESTIONS,
                      self.NUM_COLORS + self.NUM_QUESTIONS), dtype=np.bool)

        for i, obj in enumerate(objects):
            v = np.zeros(self.NUM_COLORS)
            v[obj.color] = True
            Q[i * self.NUM_QUESTIONS:(i + 1) * self.NUM_QUESTIONS,
              :self.NUM_COLORS] = np.tile(v, (self.NUM_QUESTIONS, 1))
            Q[i * self.NUM_QUESTIONS:(i + 1) * self.NUM_QUESTIONS,
              self.NUM_COLORS:] = np.diag(np.ones(self.NUM_QUESTIONS))

        return Q

    def generate_answer_matrix(self, objects):
        """
        Generates the answers matrix: [# of shape * # of Q, # of color + 4]


        `# of color + 4` = [color 1, color 2, ... , circle, rectangle, yes, no]

        :param objects: List of objects - abstract scene representation.
        :type objects: list

        :return: the answer matrix (``np.array``)
        """
        A = np.zeros((len(objects) * self.NUM_QUESTIONS,
                      self.NUM_COLORS + 4), dtype=np.bool)

        for i, obj in enumerate(objects):
            # Q1: circle or rectangle?
            if obj.shape:
                A[i * self.NUM_QUESTIONS, self.NUM_COLORS + 1] = True
            else:
                A[i * self.NUM_QUESTIONS, self.NUM_COLORS] = True

            # Q2: bottom?
            if obj.y > int(self.img_size / 2):
                A[i * self.NUM_QUESTIONS + 1, self.NUM_COLORS + 2] = True
            else:
                A[i * self.NUM_QUESTIONS + 1, self.NUM_COLORS + 3] = True

            # Q3: left?
            if obj.x < int(self.img_size / 2):
                A[i * self.NUM_QUESTIONS + 2, self.NUM_COLORS + 2] = True
            else:
                A[i * self.NUM_QUESTIONS + 2, self.NUM_COLORS + 3] = True

            # Calculate distances.
            distances = np.array(
                [((obj.x - other_obj.x) ** 2 + (obj.y - other_obj.y) ** 2)
                 for other_obj in objects])
            idx = distances.argsort()

            # Ids of closest and most distant objects.
            min_idx = idx[1]
            max_idx = idx[-1]

            # Q4: the shape of the nearest object
            A[i * self.NUM_QUESTIONS + 3,
              self.NUM_COLORS + objects[min_idx].shape] = True

            # Q5: the shape of the farthest object
            A[i * self.NUM_QUESTIONS + 4,
              self.NUM_COLORS + objects[max_idx].shape] = True

            # Q6: the color of the nearest object
            A[i * self.NUM_QUESTIONS + 5, objects[min_idx].color] = True

            # Q7: the color of the farthest object
            A[i * self.NUM_QUESTIONS + 6, objects[max_idx].color] = True

        return A

    def show_sample(self, data_dict, sample=0):
        """

        Show a sample of the current DataDict.

        :param data_dict: DataDict({'images','questions', 'targets', 'targets_index', 'scenes_description'})
        :type data_dict: DataDict

        :param sample: sample index to visualize.
        :type sample: int
        """
        import matplotlib.pyplot as plt

        # Unpack data_dict.
        images, questions, targets, targets_index, scenes_description = data_dict.values()

        # Get sample.
        image = images[sample].numpy().transpose(2, 1, 0)
        question = questions[sample].numpy()
        answer = targets_index[sample].numpy()

        # Print scene description.
        self.logger.info("Scene description :\n {}".format(scenes_description[sample]))
        self.logger.info("Question :\n {} ({})".format(question, self.question2str(question)))
        self.logger.info("Answer :\n {} ({})".format(answer, self.answer2str(answer)))

        # Generate figure.
        fig = plt.figure(1)
        plt.title('Q: {}'.format(self.question2str(question)))
        plt.xlabel('A: {}'.format(self.answer2str(answer)))
        plt.imshow(image, interpolation='nearest', aspect='auto')
        # Plot!
        plt.show()

    def plot_preprocessing(self, data_dict, logits):
        """
        Allows for some data preprocessing before the model creates a plot for
        visualization during training or inference. To be redefined in
        inheriting classes.

        :param data_dict: DataDict({'images','questions', 'targets', 'targets_index', 'scenes_description'})

        :param logits: Predictions of the model.
        :type logits: Tensor

        :return: data_tuplem aux_tuple, logits after preprocessing.

        """
        # move DataDict to cpu and detach it from the graph
        data_dict = data_dict.cpu().detach().numpy()

        # Unpack data_dict.
        images, questions, targets, targets_index, scenes_description = data_dict.values()
        batch_size = targets.shape[0]

        logits = logits.cpu().detach().numpy()

        # Convert to string
        answers_string = [self.answer2str(targets_index[batch_num]) for batch_num in range(batch_size)]
        questions_string = [self.question2str(questions[batch_num]) for batch_num in range(batch_size)]
        prediction = [self.answer2str(np.argmax(logits[batch_num])) for batch_num in range(batch_size)]

        data_dict['targets_string'] = answers_string
        data_dict['questions_string'] = questions_string

        return data_dict, prediction


if __name__ == "__main__":
    """ Tests sort of CLEVR - generates and displays a sample"""

    # "Loaded parameters".
    from utils.param_interface import ParamInterface 
    params = ParamInterface()
    params.add_default_params({'data_folder': '~/data/sort-of-clevr/',
                               'split': 'train',
                               'regenerate': False,
                               'dataset_size': 10000,
                               'img_size': 128})

    # create problem
    sortofclevr = SortOfCLEVR(params)

    batch_size = 64
    print('Number of episodes to run to cover the set once: {}'.format(sortofclevr.get_epoch_size(batch_size)))

    # get a sample
    #sample = sortofclevr[0]
    #print(repr(sample))
    #print('__getitem__ works.')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data.dataloader import DataLoader

    dataloader = DataLoader(dataset=sortofclevr, collate_fn=sortofclevr.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=8)

    # try to see if there is a speed up when generating batches w/ multiple workers
    import time

    s = time.time()
    for i, batch in enumerate(dataloader):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(dataloader.num_workers))
    print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time() - s))

    # Display single sample (0) from batch.
    #batch = next(iter(dataloader))
    #sortofclevr.show_sample(batch, 0)

    print('Unit test completed')