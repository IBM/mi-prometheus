import torch
from sequential_vision_problem import SequentialVisionProblem
from sequential_vision_problem import AuxTuple
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


@SequentialVisionProblem.register
class SequentialPixelMnist(SequentialVisionProblem):
    """
    Class generating sequences sequential mnist
    """

    def __init__(self, params):
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']
        self.num_rows = 28
        self.num_columns = 28

        self.gpu = False
        self.datasets_folder = '~/data/mnist'

        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu else {}

        # define transforms
        train_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Lambda(lambda x: x.view(1, -1, 1))])

        # load the datasets
        self.train_datasets = datasets.MNIST(self.datasets_folder, train=True, download=True,
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
        mask = torch.zeros(self.num_rows * self.num_columns)
        mask[-1] = 1

        # train_loader a generator: (data, label)
        return next(train_loader), AuxTuple(mask.type(torch.uint8))

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'batch_size': 3, 'start_index': 0, 'stop_index': 54999}
    # Create problem object.
    problem = SequentialPixelMnist(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    num_rows = 28
    num_columns = 28
    sample_num = 0
    data_tuple, _ = next(generator)
    x, y = data_tuple

    print(x[sample_num, 0].size())

    # Display single sample (0) from batch.
    problem.show_sample(x[sample_num, 0].reshape(num_rows, num_columns), y)
