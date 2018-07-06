import torch.nn as nn
import torch.nn.functional as F


class StackedAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(StackedAttention, self).__init__()


class Attention(nn.Module):
    def __init__(self, question_encoding_size, image_encoding_size, image_text_features=512):
        super(Attention, self).__init__()
        # fully connected layer to construct the key
        self.ff_image = nn.Linear(image_encoding_size, image_text_features)
        # fully connected layer to construct the query
        self.ff_ques = nn.Linear(question_encoding_size, image_text_features)
        # fully connected layer to construct the attention from the query and key
        self.ff_attention = nn.Linear(image_text_features, 1)

    def forward(self, encoded_image, encoded_question):
        # Flatten the two last dimensions of the image
        encoded_image = encoded_image.view(encoded_image.size(0), encoded_image.size(1), encoded_image.size(2)*encoded_image.size(3))

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