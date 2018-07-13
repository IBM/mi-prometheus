import torch.nn as nn
import torch.nn.functional as F


class StackedAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(StackedAttention, self).__init__()


class Attention(nn.Module):
    def __init__(self, question_image_encoding_size, key_query_size=512):
        super(Attention, self).__init__()
        # fully connected layer to construct the key
        self.ff_image = nn.Linear(question_image_encoding_size, key_query_size)
        # fully connected layer to construct the query
        self.ff_ques = nn.Linear(question_image_encoding_size, key_query_size)
        # fully connected layer to construct the attention from the query and key
        self.ff_attention = nn.Linear(key_query_size, 1)

    def forward(self, encoded_image, encoded_question):

        # Get the key
        key = self.ff_image(encoded_image)

        # Get the query, unsqueeze to be able to add the query to all channels
        query = self.ff_ques(encoded_question).unsqueeze(dim=1)
        ha = F.tanh(key + query)

        # Get attention over the different layers
        ha = self.ff_attention(ha)
        pi = F.softmax(ha, dim=-2)

        # sum the weighted channels
        vi_attended = (pi * encoded_image).sum(dim=1)

        return vi_attended