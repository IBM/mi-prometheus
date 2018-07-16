import torch.nn as nn
import torch.nn.functional as F


class ImageEncoding(nn.Module):
    def __init__(self):
        super(ImageEncoding, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.batchNorm4 = nn.BatchNorm2d(256)

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

        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)

        return x