import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..')) 

from models.model import Model
from problems.problem import DataTuple
from misc.app_state import AppState


class SimpleConvNet(Model):
    def __init__(self, params):
        super(SimpleConvNet, self).__init__(params)

        # retrieve parameters from the yaml file
        # cnn parameters
        self.depth_conv1 = params['depth_conv1']
        self.depth_conv2 = params['depth_conv2']
        self.filter_size_conv1 = params['filter_size_conv1']
        self.filter_size_conv2 = params['filter_size_conv2']
        self.num_pooling = params['num_pooling']

        # image size
        self.num_channels = params['num_channels']
        self.height = params['height']
        self.width = params['width']
        self.padding = params['padding']

        self.height_padded = self.height + sum(self.padding[0:2])
        self.width_padded = self.width + sum(self.padding[2:4])

        # Input size of the first fully connected layer:
        # We can compute the spatial size of the output volume as a function of the input volume size (W),
        # the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S),
        # and the amount of zero padding used (P) on the border. You can convince yourself that the correct formula
        # for calculating how many neurons “fit” is given by  (W−F+2P)/S+1.

        # TODO: for now we assume that padding = 0 and stride = 1
        self.width_features_conv1 = np.floor(((self.width_padded-self.filter_size_conv1) + 1)/ self.num_pooling)
        self.height_features_conv1 = np.floor(((self.height_padded-self.filter_size_conv1) + 1)/ self.num_pooling)

        self.width_features_conv2 = np.floor(((self.width_features_conv1-self.filter_size_conv2) + 1)/ self.num_pooling)
        self.height_features_conv2 = np.floor(((self.height_features_conv1-self.filter_size_conv2) + 1)/ self.num_pooling)

        self.conv1 = nn.Conv2d(self.num_channels, self.depth_conv1, kernel_size=self.filter_size_conv1)
        self.conv2 = nn.Conv2d(self.depth_conv1, self.depth_conv2, kernel_size=self.filter_size_conv2)
        self.fc1 = nn.Linear(self.depth_conv2 * self.width_features_conv2 * self.height_features_conv2, 50)
        self.fc2 = nn.Linear(50, 10)

        if self.app_state.visualize:
            self.output_conv1 = []
            self.output_conv2 = []

    def forward(self, data_tuple):

        (inputs, targets) = data_tuple

        x1 = self.conv1(inputs)
        x1_max_pool = F.relu(F.max_pool2d(x1, self.num_pooling))

        x2 = self.conv2(x1_max_pool)
        x2_max_pool = F.relu(F.max_pool2d(x2, self.num_pooling))

        x = x2_max_pool.view(-1, self.depth_conv2 * self.width_features_conv2 * self.height_features_conv2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def plot(self, data_tuple, predictions, sample_number = 0):
        """ 
        Simple plot - shows MNIST image with target and actual predicted class.

        :param data_tuple: Data tuple containing input and target batches.
        :param predictions: Prediction.
        :param sample_number: Number of sample in batch (DEFAULT: 0) 
        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False
        import matplotlib.pyplot as plt

        # Unpack tuples.
        images, targets = data_tuple

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        prediction = predictions[sample_number].cpu().detach().numpy()
 
        # Reshape image.
        if (image.shape[0] == 1):
            # This is single channel image - get rid of that dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2, according to matplotilb doc it should be ok 
            # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))    
            image = image.transpose(1, 2, 0)

        # Show data.
        plt.title('Prediction: {} (Target: {})'.format(np.argmax(prediction), target) )
        plt.imshow(image, interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()


if __name__ == '__main__':
    # Set visualization.
    AppState().visualize = True

    # Test base model.
    params = {'depth_conv1': 10, 'depth_conv2': 20, 'filter_size_conv1': 5, 'filter_size_conv2': 5, 'num_pooling': 2,
    'num_channels': 1, 'height': 28, 'width': 28, 'padding': (0,0,0,0)}

    # model
    model = SimpleConvNet(params)

    while True:
        # Generate new sequence.
        # "Image" - batch x channels x width x height
        input_np = np.random.binomial(1, 0.5, (1, 1, 28,  28))
        input = torch.from_numpy(input_np).type(torch.FloatTensor)
        # Target.
        target = torch.randint(10, (10,), dtype=torch.int64)

        dt = DataTuple(input, target)
        # prediction.
        prediction = model(dt)

        #prediction_np = np.random.binomial(1, 0.5, (1, 10))
        #prediction = torch.from_numpy(prediction_np).type(torch.FloatTensor)

        # Plot it and check whether window was closed or not. 
        if model.plot(dt, prediction):
            break
