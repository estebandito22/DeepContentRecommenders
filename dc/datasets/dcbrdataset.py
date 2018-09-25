"""Datasets for training models in PyTorch."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dc.dcbr.cf.datahandler import CFDataHandler


class DCBRDataset(Dataset):

    """Class to load dataset required for training DCBR model."""

    def __init__(self, metadata_csv, item_index=None, split='train',
                 transform=None, excluded_ids=None, data_type='scatter'):
        """
        Initialize MSDDataset.

        Args
            metadata_csv: Path to csv with the metadata for audio files to be
                          loaded.
            item_index: lookup table from itemID -> index in item_user matrix.
            split: Which split of the data to load.  'train', 'val' or 'test'.
            transform: A method that transforms the sample.
            test_small: Boolean to use only the first 32 samples in the
                        metadata_csv.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        self.metadata_csv = metadata_csv
        self.item_index = item_index
        self.split = split
        self.transform = transform
        self.excluded_ids = excluded_ids
        self.data_type = data_type

        # create metadata dataframe
        self.metadata = pd.read_csv(metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(self.excluded_ids)]
            self.metadata = self.metadata.reset_index(drop=True)

        # create train and val and test splits
        np.random.seed(10)
        train_mask = np.random.rand(self.metadata.shape[0]) < 0.90
        np.random.seed(10)
        val_mask = np.random.rand(sum(train_mask)) < 0.2/0.9

        if self.split == 'train':
            self.metadata = self.metadata[train_mask]
            self.metadata = self.metadata[~val_mask]
        elif self.split == 'val':
            self.metadata = self.metadata[train_mask]
            self.metadata = self.metadata[val_mask]
        elif self.split == 'test':
            self.metadata = self.metadata[~train_mask]
        elif self.split != 'all':
            raise ValueError("split must be: 'train', 'val', 'test', or 'all'")

    @staticmethod
    def _sample(X, length):
        rand_start = np.random.randint(0, X.size()[0] - length)
        X = X[rand_start:rand_start + length]
        return X

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        """Return sample idx from the dataset."""
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].iloc[idx])
            X = self._sample(X, 17)
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].iloc[idx])
            X = X.t()
            X = self._sample(X, 44)

        y = self.item_index[self.metadata['song_id'].iloc[idx]]

        sample = {'data': X, 'target_index': y}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DCBRPredset(Dataset):

    """Class to load dataset required for predictions with DCBR model."""

    def __init__(self, metadata_csv, dh, split='train', transform=None,
                 excluded_ids=None):
        """
        Initialize MSDDataset.

        Args
            metadata_csv: Path to csv with the metadata for audio files to be
                          loaded.
            dh: CFDataHandler object.
            split: Which split of the data to load.  'train', 'val' or 'test'.
            transform: A method that transforms the sample.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        self.metadata_csv = metadata_csv
        self.dh = dh
        self.split = split
        self.transform = transform
        self.excluded_ids = excluded_ids

        # create metadata dataframe
        self.metadata = pd.read_csv(metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(self.excluded_ids)]
            self.metadata = self.metadata.reset_index(drop=True)

        # create train and val and test splits
        if self.split == 'train':
            self.pred_item_user = dh.item_user_train
            self.filt_item_user = dh.item_user_test
        elif self.split == 'test':
            self.pred_item_user = dh.item_user_test
            self.filt_item_user = dh.item_user_train

        # create user and item lookups
        self.itemindex2songid = {v: k for (k, v) in dh.item_index.items()}

        self.metadata_user = self.metadata
        self.targets_user = None
        self.user_index = None
        self.user_songids = None
        self.user_has_songs = False

    def create_user_data(self, user):
        """
        Create metadata file for all songs in prediction set less other songs.

        Args
            i: user index.
        """
        self.user_index = self.dh.user_index[user]

        user_songs_idxs = self.pred_item_user.getcol(
            self.user_index).nonzero()[0]
        user_songs = [self.itemindex2songid[j] for j in user_songs_idxs]
        songs_mask = self.metadata['song_id'].isin(user_songs)
        self.user_songids = self.metadata['song_id'][songs_mask]

        if not self.user_songids.empty:
            self.user_has_songs = True
        else:
            self.user_has_songs = False

        user_filt_idxs = self.filt_item_user.getcol(
            self.user_index).nonzero()[0]
        user_filt = [self.itemindex2songid[j] for j in user_filt_idxs]
        filt_mask = ~self.metadata['song_id'].isin(user_filt)

        self.metadata_user = self.metadata[songs_mask | filt_mask]
        self.targets_user = np.where(self.metadata_user[
            'song_id'].isin(self.user_songids), 1.0, 0.0)

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata_user.shape[0]

    def __getitem__(self, i):
        """Return sample idx from the dataset."""
        user_index = self.user_index
        item_index = self.dh.item_index[self.metadata_user['song_id'].iloc[i]]
        y = torch.tensor(self.targets_user[i]).float()

        sample = {'user_index': user_index,
                  'item_index': item_index,
                  'target': y}

        return sample
