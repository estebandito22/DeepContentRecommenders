"""Datasets for training models in PyTorch."""

import torch


class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """Return a pytorch tensor sample."""
        inputs, targets = sample['data'], sample['target']

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).float()

        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets).float()

        return {'data': inputs, 'target': targets}


class SubtractMean(object):

    """Subtract mean from audio input."""

    def __init__(self, data_type):
        """Initialize SubtractMean."""
        if data_type == 'mel' or data_type == 'scatter':
            self.data_type = data_type
        else:
            raise ValueError("data_type must be 'mel' or 'scatter'.")

    def __call__(self, sample):
        """Subtract the appropriate mean from the sample data."""
        if self.data_type == 'mel':
            mean = 2.3779549598693848
        elif self.data_type == 'scatter':
            mean = 0.21285544335842133

        if 'data' in sample:
            key = 'data'
            inputs = sample[key]
            inputs -= mean
            sample[key] = inputs

        if 'X' in sample:
            key = 'X'
            inputs = sample[key]
            inputs -= mean
            sample[key] = inputs

        return sample
