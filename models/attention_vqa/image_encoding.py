import torch.nn as nn
import torch.nn.functional as F


class ImageEncoding(nn.Module):

    def __init__(self, num_channels_image, depth_conv1, depth_conv2, depth_conv3):
        super(ImageEncoding, self).__init__()
        # Retrieve parameters
        self.num_channels = num_channels_image
        self.depth_conv1 = depth_conv1
        self.depth_conv2 = depth_conv2
        self.depth_conv3 = depth_conv3
        self.num_pooling = 2

        # Instantiate conv layers
        self.conv1 = nn.Conv2d(self.num_channels, self.depth_conv1, kernel_size=3)
        self.conv2 = nn.Conv2d(self.depth_conv1, self.depth_conv2, kernel_size=3)
        self.conv3 = nn.Conv2d(self.depth_conv2, self.depth_conv3, kernel_size=3)

    def forward(self, image):
        # apply convectional layer 1
        x1 = self.conv1(image)

        # apply max_pooling and relu
        x1_max_pool = F.relu(F.max_pool2d(x1, self.num_pooling))

        # apply Convolutional layer 1
        x2 = self.conv2(x1_max_pool)

        # apply max_pooling and relu
        x2_max_pool = F.relu(F.max_pool2d(x2, self.num_pooling))

        # apply convectional layer 2
        x3 = self.conv3(x2_max_pool)

        # apply max_pooling and relu
        x3_max_pool = F.relu(F.max_pool2d(x3, self.num_pooling))

        return x3_max_pool


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(512)

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)

        return x