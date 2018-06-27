import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..')) 

from problems.problem import DataTuple, LabelAuxTuple
from problems.image_to_class.image_to_class_problem import ImageToClassProblem


class CIFAR(ImageToClassProblem):
    """
    Classic CFIAR classification problem.
    """

    def __init__(self, params):
        """
        Initializes CIFAR problem, calls base class initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration file).
        """

        # Call base class constructors.
        super(CIFAR, self).__init__(params)

        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']
        self.use_train_data = params['use_train_data']
        self.datasets_folder = params['folder']
        self.padding = params['padding']

        # define transforms
        train_transform = transforms.Compose([
            transforms.ToTensor()])

        # load the datasets
        self.train_datasets = datasets.CIFAR10(self.datasets_folder, train=self.use_train_data, download=True,
                                     transform=train_transform)

        # set split data (for training and validation data)
        num_train = len(self.train_datasets)

        indices = list(range(num_train))
        idx = indices[self.start_index: self.stop_index]
        self.sampler = SubsetRandomSampler(idx)

        # Class names.
        self.cifar_class_names = 'Airplane Automobile Bird Cat Deer Dog Frog Horse Shipe Truck'.split(' ')

    def generate_batch(self):

        # data loader
        train_loader = torch.utils.data.DataLoader(self.train_datasets, batch_size=self.batch_size,
                                                   sampler=self.sampler)

        # create an iterator
        train_loader = iter(train_loader)

        # train_loader a generator: (data, label)
        (data, label) = next(train_loader)

        # padding data
        data_padded = F.pad(data, self.padding, 'constant', 0)

        # Generate labels for aux tuple
        class_names = [self.cifar_class_names[i] for i in label]

        # Return DataTuple(!) and an empty (aux) tuple.
        return DataTuple(data_padded, label), LabelAuxTuple(class_names)


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'batch_size':2, 'start_index': 0, 'stop_index': 40000, 'use_train_data': True, 'folder': '~/data/cifar'}

    # Create problem object.
    problem = CIFAR(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    dt, at = next(generator)

    # Display single sample (0) from batch.
    problem.show_sample(dt, at, 0)
