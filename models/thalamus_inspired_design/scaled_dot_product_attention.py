import torch
from torch import nn
import numpy as np

# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..'))

from models.model import Model
from problems.problem import DataTuple
from misc.app_state import AppState


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, params):
        super(ScaledDotProductAttention, self).__init__()

        self.image_depth = params['image_depth']
        self.image_height = params['image_height']
        self.image_width = params['image_width']
        self.num_words = params['image_width']

        self.hidden_size = 10

        self.fc_key = nn.Linear(self.image_depth*self.image_height, self.hidden_size)
        self.fc_query = nn.Linear(self.num_words, self.hidden_size)

        self.temper = np.power(d_model, 0.5)
        self.softmax = nn.Softmax()

    def forward(self, image, question):

        # step 1: Get the Key and the Query
        Key = self.fc_key(question)
        Query = self.fc_query(image)

        # step 2:


        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper


        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


if __name__ == "__main__":
    attention = ScaledDotProductAttention()

    # Query
    q = torch.randn()

    # Key