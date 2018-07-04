import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from models.attention_vqa.image_encoding import ImageEncoding
from models.attention_vqa.attention import StackedAttention
from models.model import Model
from misc.app_state import AppState


class AttentionVQA(Model):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, params):
        super(AttentionVQA, self).__init__(params)

        # Retrieve attention and image parameters
        question_features = 10
        self.num_channels_image = 3
        self.glimpses = 3
        self.mid_features = 64

        # LSTM parameters
        self.hidden_size = 10
        self.word_embedded_size = 7
        self.num_layers = 3

        # Instantiate class for image encoding
        self.image_encoding = ImageEncoding(
            num_channels_image=3,
            depth_conv1=16,
            depth_conv2=32,
            depth_conv3=64
        )

        # Instantiate class for question encoding
        self.lstm = nn.LSTM(self.word_embedded_size, self.hidden_size, self.num_layers, batch_first=True)

        # Instantiate class for attention
        self.apply_attention = StackedAttention(
            q_features=question_features,
            mid_features=self.mid_features,
            glimpses=self.glimpses,
            drop=0.5,
        )

        self.classifier = Classifier(
            in_features=self.glimpses * self.mid_features + question_features,
            mid_features=1024,
            out_features=10,
            drop=0.5,
        )

    def forward(self, data_tuple):
        (images, questions), _ = data_tuple
        images = images.transpose(1, 3)
        images = images.transpose(2, 3)

        # step1 : encode image
        encoded_images = self.image_encoding(images)

        # Initial hidden_state for question encoding
        batch_size = images.size(0)
        hx, cx = self.init_hidden_states(batch_size)

        # step2 : encode question
        encoded_question, _ = self.lstm(questions, (hx, cx))
        last_layer_encoded_question = encoded_question[:, -1, :]

        # step3 : apply attention
        encoded_image_attention = self.apply_attention(encoded_images, last_layer_encoded_question)

        # step 4: classifying based in the encoded questions and attention
        combined = torch.cat([encoded_image_attention, last_layer_encoded_question], dim=1)
        answer = self.classifier(combined)

        return answer

    def init_hidden_states(self, batch_size):
        dtype = AppState().dtype
        hx = torch.randn(self.num_layers, batch_size, self.hidden_size).type(dtype)
        cx = torch.randn(self.num_layers, batch_size, self.hidden_size).type(dtype)

        return hx, cx

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


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


if __name__ == '__main__':
    # Set visualization.
    AppState().visualize = True

    # Test base model.
    params = []

    # model
    model = AttentionVQA(params)

    while True:
        # Generate new sequence.
        # "Image" - batch x channels x width x height
        input_np = np.random.binomial(1, 0.5, (1, 128,  128, 3))
        image = torch.from_numpy(input_np).type(torch.FloatTensor)

        #Question
        questions_np = np.random.binomial(1, 0.5, (1, 13, 7))
        questions = torch.from_numpy(questions_np).type(torch.FloatTensor)

        # Target.
        target = torch.randint(10, (10,), dtype=torch.int64)

        dt = (image, questions), target
        # prediction.
        prediction = model(dt)

        # Plot it and check whether window was closed or not.
        if model.plot(dt, prediction):
            break