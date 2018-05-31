import torch
from vision_problem import VisionProblem
from vision_problem import DataTuple
from torchvision import datasets, transforms


@VisionProblem.register
class SequentialMnist(VisionProblem):
    """
    Class generating sequences sequential mnist
    """

    def __init__(self, params):
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.gpu = False
        self.datasets_folder = '/data_mnist'

    def generate_batch(self):

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu else {}

        # load train datasets
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.datasets_folder, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x.view(-1, 1))
                           ])),
            batch_size=self.batch_size, shuffle=True, **kwargs)

        return train_loader

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'batch_size':1}
    # Create problem object.
    problem = SequentialMnist(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    num_rows = 28
    num_columns = 28
    #TODO: fix batch problem with view
    for x, y in generator:
        # Print single sample (0) from batch.
        print('data:', x)
        print('label:', y)

        #problem.show_sample(x.view(num_rows, num_columns), y)
