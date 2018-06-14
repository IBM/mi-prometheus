# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..','..')) 

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from problems.problem import DataTuple, MaskAuxTuple
from problems.video_to_class.video_to_class_problem import VideoToClassProblem

class SequentialRowMNIST(VideoToClassProblem):
    """
    Class generating sequences sequential mnist
    """

    def __init__(self, params):
        super(SequentialRowMNIST, self).__init__(params)
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']
        self.use_train_data = params['use_train_data']
        self.num_rows = 28
        self.num_columns = 28
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
        # set split
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

        # create mask
        mask = torch.zeros(self.num_rows)
        mask[-1] = 1

        # train_loader a generator: (data, label)
        (data, label) = next(train_loader)

        # Return DataTuple(!) and an empty (aux) tuple.
        return DataTuple(data,label), MaskAuxTuple(mask.type(torch.uint8))

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'batch_size': 10, 'start_index': 0, 'stop_index': 54999, 'use_train_data': True, 'mnist_folder': '~/data/mnist'}

    # Create problem object.
    problem = SequentialRowMNIST(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    num_rows = 28
    num_columns = 28
    sample_num = 0
    data_tuple, _ = next(generator)
    x, y = data_tuple

    print(x.size())

    # Display single sample (0) from batch.
    problem.show_sample(x[sample_num, 0], y)
