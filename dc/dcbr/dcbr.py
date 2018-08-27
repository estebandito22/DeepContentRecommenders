"""Class for DCBR Model."""

import torch.nn as nn

from dc.dcbr.nn.audiomodel import ConvNetMel
from dc.dcbr.nn.audiomodel import ConvNetScatter


class DCBRNet(nn.Module):

    """Simple class for DCBR network."""

    def __init__(self, dict_args):
        """
        Initialize DCBR network.

        dict_args: dictionary of arguments including:
            'output_size': output size of the convnet.
            'dropout': dropout in convnet.
            'init_fc_bias': initizlization of convnet output layer bias.
            'bn_momentum': batch normalization momentum.
            'data_type': 'mel' or 'scatter'.
        """
        super(DCBRNet, self).__init__()

        self.dict_args = dict_args
        self.data_type = dict_args['data_type']

        if self.data_type == 'mel':
            self.conv = ConvNetMel(self.dict_args)
        elif self.data_type == 'scatter':
            self.conv = ConvNetScatter(self.dict_args)

    def forward(self, x):
        """Forward pass."""
        return self.conv(x)
