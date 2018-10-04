"""Datasets for training models in PyTorch."""

import random
import numpy as np
import torch

from dc.datasets.dcueitemset import DCUEItemset


class DCUELMItemset(DCUEItemset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets_txt, metadata_csv, data_type='mel',
                 n_users=20000, n_items=10000, transform=None,
                 excluded_ids=None, random_seed=None):
        """
        Initialize DCUE dataset.

        Args
            triplets_txt: path to text file with triplets of fields.
                user_id, song_id, score.
            metadata_csv: Path to csv with the metadata for audio files to be
                loaded.
            data_type: 'scatter' or 'mel'.
            split: 'train', 'val' or 'test'.
            transform: A method that transforms the sample.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        DCUEItemset.__init__(
            self, triplets_txt, metadata_csv, data_type, n_users, n_items,
            transform, excluded_ids, random_seed)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        song_id = self.metadata.iloc[i]['song_id']
        song_idx = self.songid2metaindex[song_id]

        # load torch positive tensor
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].loc[song_idx])
            X = self._sample(X, 17, 0)
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[song_idx])
            X = self._sample(X, 131, 1)

        # load text
        # t = np.loadtxt(self.metadata['bio'].iloc[song_idx])
        t = [np.random.randint(0, 20000, [20]) for i in range(5)]
        t = random.choice(t)
        t = torch.from_numpy(t).long()

        sample = {'X': X, 't': t, 'metadata_index': song_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample
