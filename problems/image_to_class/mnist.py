# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..')) 

import torch
from vision_problem import VisionProblem
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from problems.problem import DataTuple

class MNIST(VisionProblem):
    """
    Classic MNIST classification problem.
    """

    def __init__(self, params):
        """
        Initializes MNIST problem, calls base class initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration file).
        """

        # Call base class constructors.
        super(MNIST, self).__init__(params)

        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']
        self.use_train_data = params['use_train_data']
        self.datasets_folder = params['mnist_folder']
        # TODO: WHY?? Fix this!
        self.gpu = False

        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu else {}

        # define transforms
        train_transform = transforms.Compose([
            transforms.ToTensor()])

        # load the datasets
        self.train_datasets = datasets.MNIST(self.datasets_folder, train=self.use_train_data, download=True,
                                     transform=train_transform)

        # set split data (for training and validation data)
        num_train = len(self.train_datasets)
        indices = list(range(num_train))
        idx = indices[self.start_index: self.stop_index]
        self.sampler = SubsetRandomSampler(idx)

    def generate_batch(self):

        # data loader
        train_loader = torch.utils.data.DataLoader(self.train_datasets, batch_size=self.batch_size,
                                                   sampler=self.sampler, **self.kwargs)
        # create an iterator
        train_loader = iter(train_loader)

        # train_loader a generator: (data, label)
        (data, label) = next(train_loader)

        # Return DataTuple(!) and an empty (aux) tuple.
        return DataTuple(data,label), ()


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'batch_size':1, 'start_index': 0, 'stop_index': 54999, 'use_train_data': True, 'mnist_folder': '~/data/mnist'}
    # Create problem object.
    problem = MNIST(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple, y = next(generator)
    x, y = data_tuple

    # Display single sample (0) from batch.
    sample_numer = 0
    problem.show_sample(x[sample_numer, 0], y)
