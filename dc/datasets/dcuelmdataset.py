"""Datasets for training models in PyTorch."""

import random
import numpy as np
import torch

from dc.datasets.dcuedataset import DCUEDataset


class DCUELMDataset(DCUEDataset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets_txt, metadata_csv, neg_samples=None,
                 split='train', data_type='mel', n_users=20000, n_items=10000,
                 transform=None, excluded_ids=None, random_seed=None):
        """
        Initialize DCUELM dataset.

        Args
            triplets_txt: path to text file with triplets of fields.
                user_id, song_id, score.
            metadata_csv: Path to csv with the metadata for audio files to be
                loaded.
            neg_samples: The number of negative samples to draw.
            data_type: 'scatter' or 'mel'.
            split: 'train', 'val' or 'test'.
            transform: A method that transforms the sample.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        DCUEDataset.__init__(
            self, triplets_txt, metadata_csv, neg_samples, split, data_type,
            n_users, n_items, transform, excluded_ids, random_seed)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        song_id = self.dh.triplets_df.iloc[i]['song_id']
        song_idx = self.songid2metaindex[song_id]

        # load torch positive tensor
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].loc[song_idx])
            X = self._sample(X, 17, 0)
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[song_idx])
            X = self._sample(X, 131, 1)

        # returned for user embedding
        user_idx = self.dh.user_index[self.dh.triplets_df.iloc[i]['user_id']]
        user_idx = torch.tensor(user_idx)

        # all targets are -1
        y = torch.tensor((), dtype=torch.float32).new_full(
            [self.neg_samples], -1)

        # load text
        # t = np.loadtxt(self.metadata['bio'].iloc[song_idx])
        t = [np.random.randint(0, 20000, [20]) for i in range(5)]
        t = random.choice(t)
        t = torch.from_numpy(t).long()

        sample = {'u': user_idx, 'X': X, 'y': y, 't': t}

        if self.transform:
            sample = self.transform(sample)

        return sample
