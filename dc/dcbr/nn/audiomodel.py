"""PyTorch classes for the audio model component in DCBR."""

from torch import nn
import numpy as np


class ConvNetScatter(nn.Module):

    """ConvNet used on data prepared with scattering transform."""

    def __init__(self, dict_args):
        """
        Initialize ConvNet used on data prepared with scattering transform.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(ConvNetScatter, self).__init__()
        self.output_size = dict_args["output_size"]
        self.bn_momentum = dict_args["bn_momentum"]
        # input_size = batch size x 1 x 441 x 17
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=128, kernel_size=(1, 11),
            stride=(1, 1), padding=(0, 5), bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(441, 1))
        self.batchnorm1 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu1 = nn.ReLU()
        # batch size x 128 x 1 x 17

        self.layer2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 11),
            stride=(1, 1), padding=(0, 5), bias=False)
        self.pool2 = nn.MaxPool2d(
            kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.batchnorm2 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu2 = nn.ReLU()
        # batch size x 128 x 1 x 17

        self.layer3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 11),
            stride=(1, 1), bias=False)
        self.pool3 = nn.MaxPool2d(
            kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.batchnorm3 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu3 = nn.ReLU()
        # batch size x 128 x 1 x 7

        self.layer4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 4),
            stride=(1, 1), bias=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.batchnorm4 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu4 = nn.ReLU()
        # batch size x 128 x 1 x 2

        self.layer5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(1, 1),
            stride=(1, 1), bias=False)
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 2))
        self.batchnorm5 = nn.BatchNorm2d(
            256, momentum=self.bn_momentum, track_running_stats=False)
        self.relu5 = nn.ReLU()
        # batch size x 1 x 256 x 1

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
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.layer5(x)
        x = self.pool5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)

        return self.fc(x.view(-1, 256))


class ConvNetMel(nn.Module):

    """ConvNet used on data prepared with melspectogram transform."""

    def __init__(self, dict_args):
        """
        Initialize ConvNet used on data prepared with scattering transform.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(ConvNetMel, self).__init__()
        self.output_size = dict_args["output_size"]
        self.bn_momentum = dict_args["bn_momentum"]
        # input_size = batch size x 1 x 128 x 44
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=128, kernel_size=(1, 29),
            stride=(1, 1), padding=(0, 14), bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(128, 1))
        self.batchnorm1 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu1 = nn.ReLU()
        # batch size x 128 x 1 x 44

        self.layer2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 29),
            stride=(1, 1), padding=(0, 14), bias=False)
        self.pool2 = nn.MaxPool2d(
            kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.batchnorm2 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu2 = nn.ReLU()
        # batch size x 128 x 1 x 44

        self.layer3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 29),
            stride=(1, 1), bias=False)
        self.pool3 = nn.MaxPool2d(
            kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.batchnorm3 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu3 = nn.ReLU()
        # batch size x 128 x 1 x 16

        self.layer4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 10),
            stride=(1, 1), bias=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.batchnorm4 = nn.BatchNorm2d(
            128, momentum=self.bn_momentum, track_running_stats=False)
        self.relu4 = nn.ReLU()
        # batch size x 128 x 1 x 3

        self.layer5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(1, 1),
            stride=(1, 1), bias=False)
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 3))
        self.batchnorm5 = nn.BatchNorm2d(
            256, momentum=self.bn_momentum, track_running_stats=False)
        self.relu5 = nn.ReLU()
        # batch size x 1 x 256 x 1

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
        x = self.layer1(x)
        x = self.pool1(x)
        # x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        # x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        # x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        # x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.layer5(x)
        x = self.pool5(x)
        # x = self.batchnorm5(x)
        x = self.relu5(x)

        return self.fc(x.view(-1, 256))
