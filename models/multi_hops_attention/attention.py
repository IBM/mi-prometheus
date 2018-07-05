import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

from models.model import Model
from problems.problem import DataTuple
from misc.app_state import AppState


class StackedAttention(nn.Module):
    def __init__(self, q_features, mid_features, glimpses, drop=0.0):
        super(StackedAttention, self).__init__()
        self.key_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)

    def forward(self, encoded_image, encoded_question):
        key = self.key_lin(self.drop(encoded_question))
        key_expanded = tile_2d_over_nd(key, encoded_image)

        x = F.relu(encoded_image + key_expanded)
        attention = self.x_conv(self.drop(x))

        x = self.apply_attention(encoded_image, attention)

        return x

    def apply_attention(self, encoded_image, attention):
        """ Apply any number of attention maps over the encoded_image.
            The attention map has to have the same size in all dimensions except dim=1.
        """
        n, c = encoded_image.size()[:2]
        glimpses = attention.size(1)

        # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
        encoded_image = encoded_image.reshape(n, c, -1)
        attention = attention.reshape(n, glimpses, -1)

        s = encoded_image.size(2)

        # apply a softmax to each attention map separately
        # since softmax only takes 2d encoded_images, we have to collapse the first two dimensions together
        # so that each glimpse is normalized separately
        attention = attention.view(n * glimpses, -1)
        attention = F.softmax(attention, dim=-1)

        # apply the weighting by creating a new dim to tile both tensors over
        target_size = [n, glimpses, c, s]
        encoded_image = encoded_image.view(n, 1, c, s).expand(*target_size)
        attention = attention.view(n, glimpses, 1, s).expand(*target_size)
        weighted = encoded_image * attention
        # sum over only the spatial dimension
        weighted_mean = weighted.sum(dim=3)
        # the shape at this point is (n, glimpses, c, 1)
        return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = 2  # one feature map
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)

    return tiled