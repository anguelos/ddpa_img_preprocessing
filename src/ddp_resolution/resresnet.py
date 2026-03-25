import torch_mentor as mtr
import torch
from torch import nn
import torchvision as tv


class ResResNet(mtr.Mentee):
    def __init__(
        self,
    ):
        super().__init__()
        self.resnet = tv.models.resnet.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
