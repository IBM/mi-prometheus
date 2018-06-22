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

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, data_tuple):

        (inputs, targets) = data_tuple

        x = F.relu(F.max_pool2d(self.conv1(inputs), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
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
    params = []
    model = SimpleConvNet(params)

    while True:
        # Generate new sequence.
        # "Image" - batch x channels x width x height
        input_np = np.random.binomial(1, 0.5, (1, 1, 28,  28))
        input = torch.from_numpy(input_np).type(torch.FloatTensor)
        # Target.
        target = torch.randint(10, (10,), dtype=torch.int64)
        # prediction.
        prediction_np = np.random.binomial(1, 0.5, (1, 10))
        prediction = torch.from_numpy(prediction_np).type(torch.FloatTensor)

        dt = DataTuple(input, target)
        # Plot it and check whether window was closed or not. 
        if model.plot(dt, prediction):
            break
