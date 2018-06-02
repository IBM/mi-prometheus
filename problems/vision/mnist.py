import torch
from vision_problem import VisionProblem
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from vision_problem import DataTuple, AuxTuple

@VisionProblem.register
class Mnist(VisionProblem):
    """
    Class generating sequences sequential mnist
    """

    def __init__(self, params):
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']

        self.gpu = False
        self.datasets_folder = './data_mnist'

    def generate_batch(self):

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu else {}

        # define transforms
        train_transform = transforms.Compose([
            transforms.ToTensor()])

        # load the datasets
        train_datasets = datasets.MNIST(self.datasets_folder, train=True, download=True,
                                     transform=train_transform)
        # set split
        num_train = len(train_datasets)
        indices = list(range(num_train))

        idx = indices[self.start_index: self.stop_index]
        sampler = SubsetRandomSampler(idx)

        # loader
        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=self.batch_size,
                                                   sampler=sampler, **kwargs)
        # create an iterator
        train_loader = iter(train_loader)

        # train_loader a generator: (data, label)
        return next(train_loader), ()

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'batch_size':1, 'start_index': 0, 'stop_index': 54999}
    # Create problem object.
    problem = Mnist(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    x, y = next(generator)
    # Display single sample (0) from batch.
    sample_numer = 0
    problem.show_sample(x[sample_numer, 0], y)
