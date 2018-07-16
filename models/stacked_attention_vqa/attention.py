import torch.nn as nn
import torch.nn.functional as F


class StackedAttention(nn.Module):
    def __init__(self, question_image_encoding_size, key_query_size, num_att_layers=2):
        super(StackedAttention, self).__init__()

        self.san = nn.ModuleList(
            [Attention(question_image_encoding_size, key_query_size)] * num_att_layers)

    def forward(self, encoded_image, encoded_question):

        for att_layer in self.san:
            u = att_layer(encoded_image, encoded_question)

        return u


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
        weighted_key_query = F.tanh(key + query)

        # Get attention over the different layers
        weighted_key_query = self.ff_attention(weighted_key_query)
        attention_prob = F.softmax(weighted_key_query, dim=-2)

        # sum the weighted channels
        vi_attended = (attention_prob * encoded_image).sum(dim=1)
        u = vi_attended + encoded_question

        return u