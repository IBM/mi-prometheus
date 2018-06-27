import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..'))

from models.model import Model
from problems.problem import DataTuple
from misc.app_state import AppState


class HierarchicalCNN(Model):
    def __init__(self, params):
        super(HierarchicalCNN, self).__init__(params)

        # plug first cnn layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # plug second cnn layers
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(1760, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, data_tuple):
        # unpack data
        image, target = data_tuple

        for i in range(2):
            # apply cnn layer1
            features_layer1 = F.relu(F.max_pool2d(self.conv1(image), 2))

            # apply cnn layer2
            features_layer2 = F.relu(F.max_pool2d(self.conv2(features_layer1), 2))

            features_layer1_flatten = features_layer1.view(-1, 1440)
            features_layer2_flatten = features_layer2.view(-1, 320)

            features = torch.cat((features_layer1_flatten, features_layer2_flatten), dim=1)

        x = F.relu(self.fc1(features))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    # Set visualization.
    AppState().visualize = True

    # Test base model
    params = []
    model = HierarchicalCNN(params)

    while True:
        # Generate new sequence.
        # "Image" - batch x channels x width x height
        input_np = np.random.binomial(1, 0.5, (2, 1, 28,  28))
        input = torch.from_numpy(input_np).type(torch.FloatTensor)
        # Target.
        target = torch.randint(10, (10,), dtype=torch.int64)
        # prediction.
        prediction_np = np.random.binomial(1, 0.5, (2, 10))
        prediction = torch.from_numpy(prediction_np).type(torch.FloatTensor)

        dt = DataTuple(input, target)

        # apply model
        output = model(dt)

        # Plot it and check whether window was closed or not.
        if model.plot(dt, prediction):
            break