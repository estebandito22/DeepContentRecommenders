"""PyTorch classes for the audio model component in DCUE."""

from torch import nn
import numpy as np


class ConvNetMel1D(nn.Module):

    """ConvNet used on data prepared with melspectogram transform."""

    def __init__(self, dict_args):
        """
        Initialize ConvNetMel1D.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
                bn_momentum: momentum for batch normalization.
                dropout: dropout rate.
        """
        super(ConvNetMel1D, self).__init__()
        self.output_size = dict_args["output_size"]
        self.bn_momentum = dict_args["bn_momentum"]
        self.dropout = dict_args["dropout"]
        self.bias = True if self.dropout > 0 else False
        # input_size = batch size x 128 x 131
        self.bn0 = nn.BatchNorm1d(128, momentum=self.bn_momentum)
        self.layer1 = nn.Conv1d(
            in_channels=128, out_channels=32, kernel_size=3,
            stride=1, padding=0, bias=self.bias)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        if self.dropout > 0:
            self.drop_bn1 = nn.Dropout(self.dropout)
        else:
            self.drop_bn1 = nn.BatchNorm1d(
                32, momentum=self.bn_momentum)
        self.relu1 = nn.ReLU()
        # batch size x 32 x 64

        self.layer2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3,
            stride=1, padding=0, bias=self.bias)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        if self.dropout > 0:
            self.drop_bn2 = nn.Dropout(self.dropout)
        else:
            self.drop_bn2 = nn.BatchNorm1d(
                64, momentum=self.bn_momentum)
        self.relu2 = nn.ReLU()
        # batch size x 64 x 31

        self.layer3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3,
            stride=1, padding=0, bias=self.bias)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        if self.dropout > 0:
            self.drop_bn3 = nn.Dropout(self.dropout)
        else:
            self.drop_bn3 = nn.BatchNorm1d(
                128, momentum=self.bn_momentum)
        self.relu3 = nn.ReLU()
        # batch size x 128 x 14

        self.layer4 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3,
            stride=1, padding=0, bias=self.bias)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        if self.dropout > 0:
            self.drop_bn4 = nn.Dropout(self.dropout)
        else:
            self.drop_bn4 = nn.BatchNorm1d(
                128, momentum=self.bn_momentum)
        self.relu4 = nn.ReLU()
        # batch size x 128 x 6

        self.layer5 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=1,
            stride=1, bias=self.bias)
        self.pool5 = nn.MaxPool1d(kernel_size=4)
        if self.dropout > 0:
            self.drop_bn5 = nn.Dropout(self.dropout)
        else:
            self.drop_bn5 = nn.BatchNorm1d(
                256, momentum=self.bn_momentum)
        self.relu5 = nn.ReLU()
        # batch size x 256 x 1

        self.fc = nn.Linear(256, self.output_size)

        # initizlize weights
        nn.init.xavier_normal_(self.layer1.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer2.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer3.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer4.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer5.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.fc.weight, np.sqrt(2))

    def forward(self, x):
        """Execute forward pass."""
        x = self.bn0(x)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.drop_bn1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.drop_bn2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.drop_bn3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        x = self.drop_bn4(x)
        x = self.relu4(x)
        x = self.layer5(x)
        x = self.pool5(x)
        x = self.drop_bn5(x)
        x = self.relu5(x)

        return self.fc(x.view(-1, 256))
