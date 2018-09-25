"""PyTorch classes for the audio model component in DCUE."""

from torch import nn
import numpy as np


class L3NetMelTrunc2D(nn.Module):

    """ConvNet used on data prepared with melspectogram transform."""

    def __init__(self, dict_args):
        """
        Initialize L3NetMelTrunc.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
                bn_momentum: momentum for batch normalization.
                dropout: dropout rate.
        """
        super(L3NetMelTrunc2D, self).__init__()
        self.output_size = dict_args["output_size"]
        self.bn_momentum = dict_args["bn_momentum"]
        self.dropout = dict_args["dropout"]
        self.bias = True if self.dropout > 0 else False
        # input_size = batch size x 1 x 128 x 44
        self.bn0 = nn.BatchNorm2d(1, momentum=self.bn_momentum)
        self.layer1_1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=2,
            stride=1, padding=0, bias=self.bias)
        self.layer1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2,
            stride=1, padding=0, bias=self.bias)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        if self.dropout > 0:
            self.drop_bn1 = nn.Dropout2d(self.dropout)
        else:
            self.drop_bn1 = nn.BatchNorm2d(
                64, momentum=self.bn_momentum)
        self.relu1 = nn.ReLU()
        self.layer_outsize1 = [64, 64, 21]
        # batch size x 64 x 63 x 21

        self.layer2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=2,
            stride=1, padding=0, bias=self.bias)
        self.layer2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=2,
            stride=1, padding=0, bias=self.bias)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        if self.dropout > 0:
            self.drop_bn2 = nn.Dropout2d(self.dropout)
        else:
            self.drop_bn2 = nn.BatchNorm2d(
                128, momentum=self.bn_momentum)
        self.relu2 = nn.ReLU()
        self.layer_outsize2 = [128, 30, 9]
        # batch size x 128 x 30 x 9

        self.layer3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=2,
            stride=1, padding=0, bias=self.bias)
        self.layer3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=2,
            stride=1, padding=1, bias=self.bias)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        if self.dropout > 0:
            self.drop_bn3 = nn.Dropout2d(self.dropout)
        else:
            self.drop_bn3 = nn.BatchNorm2d(
                256, momentum=self.bn_momentum)
        self.relu3 = nn.ReLU()
        self.layer_outsize3 = [256, 14, 4]
        # batch size x 256 x 14 x 4

        self.layer4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=2,
            stride=1, padding=0, bias=self.bias)
        self.layer4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=2,
            stride=1, padding=0, bias=self.bias)
        self.pool4 = nn.MaxPool2d(kernel_size=(7, 2))
        if self.dropout > 0:
            self.drop_bn4 = nn.Dropout2d(self.dropout)
        else:
            self.drop_bn4 = nn.BatchNorm2d(
                512, momentum=self.bn_momentum)
        self.relu4 = nn.ReLU()
        self.layer_outsize4 = [512, 1, 1]
        # batch size x 512 x 1 x 1

        # initizlize weights
        nn.init.xavier_normal_(self.layer1_1.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer1_2.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer2_1.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer2_2.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer3_1.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer3_2.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer4_1.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer4_2.weight, np.sqrt(2))

    def forward(self, x):
        """Execute forward pass."""
        x = self.bn0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.pool1(x)
        x = self.drop_bn1(x)
        x = self.relu1(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.pool2(x)
        x = self.drop_bn2(x)
        x = self.relu2(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.pool3(x)
        x = self.drop_bn3(x)
        x = self.relu3(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.pool4(x)
        x = self.drop_bn4(x)
        x = self.relu4(x)

        return x.view(-1, 512)
