import torchvision.models as models
import torch

from models.model import Model


class AlexnetWrapper(Model):
    """
    Wrapper class to Alexnet model from TorchVision.
    """

    def __init__(self, params):
        super(AlexnetWrapper, self).__init__(params)

        # set model from torchvision
        self.model = models.alexnet(params["num_classes"])

    def forward(self, data_tuple):

        # get data
        (x, _) = data_tuple

        # construct the three channels needed for alexnet
        if x.size(1) != 3:
            # inputs_size = (batch_size, num_channel, numb_columns, num_rows)
            num_channel = 3
            inputs_size = (x.size(0), num_channel, x.size(2), x.size(3))
            inputs = torch.zeros(inputs_size)

            for i in range(num_channel):
                inputs[:, None, i, :, :] = x
        else:
            inputs = x

        outputs = self.model(inputs)

        return outputs
