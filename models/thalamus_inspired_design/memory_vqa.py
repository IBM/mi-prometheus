import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

from models.model import Model
from problems.problem import DataTuple
from misc.app_state import AppState


class MemoryVQA(Model):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, params):
        super(MemoryVQA, self).__init__(params)
        question_features = 10
        vision_features = 3
        glimpses = 2
        mid_features = 50
        self.hidden_size = 10
        input_size = 7
        self.num_layers = 3

        self.encoded_image = ImageEncoding(
            v_features = 3,
            mid_features = mid_features
        )

        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.attention = Attention(
            q_features=question_features,
            mid_features=mid_features,
            glimpses=glimpses,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=10,
            drop=0.5,
        )

    def forward(self, data_tuple):
        (images, questions), _ = data_tuple
        images = images.transpose(1, 3)
        images = images.transpose(2, 3)

        encoded_images = self.encoded_image(images)

        # initial hidden_state
        batch_size = images.size(0)
        hx = torch.randn(self.num_layers, batch_size, self.hidden_size)
        cx = torch.randn(self.num_layers, batch_size, self.hidden_size)

        encoded_question, _ = self.lstm(questions, (hx, cx))
        encoded_question_last = encoded_question[:, -1, :]

        a = self.attention(encoded_images, encoded_question_last)
        v = apply_attention(images, a)

        combined = torch.cat([v, encoded_question_last], dim=1)
        answer = self.classifier(combined)
        return answer

    def plot(self, data_tuple, predictions, sample_number=0):
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
        import matplotlib.gridspec as gridspec

        # Unpack tuples.
        (images, texts), targets = data_tuple

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        prediction = predictions[sample_number].cpu().detach().numpy()

        # Show data.
        plt.title('Prediction: {} (Target: {})'.format(np.argmax(prediction), target))
        plt.imshow(image.transpose(0,1,2), interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()
        exit()


class ImageEncoding(nn.Module):
    def __init__(self, v_features, mid_features, drop=0.0):
        super(ImageEncoding, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v):
        v = self.v_conv(self.drop(v))

        return v


class Attention(nn.Module):
    def __init__(self, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.reshape(n, c, -1)
    attention = attention.reshape(n, glimpses, -1)

    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention, dim=-1)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3)

    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = 2  # one feature map
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)

    return tiled

if __name__ == '__main__':
    # Set visualization.
    AppState().visualize = True

    # Test base model.
    params = []

    # model
    model = MemoryVQA(params)

    while True:
        # Generate new sequence.
        # "Image" - batch x channels x width x height
        input_np = np.random.binomial(1, 0.5, (1, 128,  128, 3))
        image = torch.from_numpy(input_np).type(torch.FloatTensor)

        #Question
        questions_np = np.random.binomial(1, 0.5, (1, 13))
        questions = torch.from_numpy(questions_np).type(torch.FloatTensor)

        # Target.
        target = torch.randint(10, (10,), dtype=torch.int64)

        dt = (image, questions), target
        # prediction.
        prediction = model(dt)

        # Plot it and check whether window was closed or not.
        if model.plot(dt, prediction):
            break