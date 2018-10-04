"""Datasets for training models in PyTorch."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dc.dcbr.cf.datahandler import CFDataHandler


class DCUEDataset(Dataset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets_txt, metadata_csv, neg_samples=None,
                 split='train', data_type='mel', n_users=20000, n_items=10000,
                 transform=None, excluded_ids=None, random_seed=None):
        """
        Initialize DCUE dataset.

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
        self.triplets_txt = triplets_txt
        self.metadata_csv = metadata_csv
        self.neg_samples = neg_samples
        self.split = split
        self.data_type = data_type
        self.transform = transform
        self.excluded_ids = excluded_ids
        self.random_seed = random_seed

        # load audio metadata dataset
        self._load_metadata()
        # load taste profile dataset
        self.dh = CFDataHandler(triplets_txt)
        # align the two datasets
        self._align_datasets(n_items, n_users)
        # build indexes to convert between datasets
        self._build_indexes()

        # build all items to sample from
        self.n_items = self.dh.item_user.shape[0]
        self.n_users = self.dh.item_user.shape[1]
        self.all_items = np.arange(0, self.n_items)

        # limit data to specific split
        self._train_test_split()

        # dataset stats
        self.uniq_songs = self.dh.triplets_df['song_id'].unique()
        self.uniq_song_idxs = [self.dh.item_index[song_id] for
                               song_id in self.uniq_songs]

    def _build_indexes(self):
        # lookup tables
        self.songid2metaindex = {v: k for (k, v)
                                 in self.metadata['song_id'].to_dict().items()}
        self.itemindex2songid = {v: k for (k, v)
                                 in self.dh.item_index.items()}

    def _align_datasets(self, n_items, n_users):
        # limit taste df to songs with audio only
        self.dh.triplets_df = self.dh.triplets_df[
            self.dh.triplets_df['song_id'].isin(self.metadata['song_id'])]

        # limit taste df to most frequent songs and users
        top_items = self.dh.triplets_df.groupby(
            'song_id').count().nlargest(n_items, 'user_id').index.values
        top_users = self.dh.triplets_df.groupby(
            'user_id').count().nlargest(n_users, 'song_id').index.values
        top_items_mask = self.dh.triplets_df['song_id'].isin(top_items)
        top_users_mask = self.dh.triplets_df['user_id'].isin(top_users)
        self.dh.triplets_df = self.dh.triplets_df[
            (top_items_mask) & (top_users_mask)]

        # build item_user sparse matrix for quick item lookups
        self.dh.item_user_matrix()

    def _load_metadata(self):
        # load metadata and exclude ids
        self.metadata = pd.read_csv(self.metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(self.excluded_ids)]
            self.metadata = self.metadata.reset_index(drop=True)

    def _train_test_split(self):
        # create train and val and test splits
        uniq_songs = self.dh.triplets_df['song_id'].unique()
        np.random.seed(10)
        song_train_mask = np.random.rand(len(uniq_songs)) < 0.80
        train_songs = uniq_songs[song_train_mask]
        np.random.seed(10)
        song_val_mask = np.random.rand(sum(song_train_mask)) < 0.1/0.8
        val_songs = train_songs[song_val_mask]

        if self.split == 'train':
            self.dh.triplets_df = self.dh.triplets_df[
                (self.dh.triplets_df['song_id'].isin(train_songs)) &
                (~self.dh.triplets_df['song_id'].isin(val_songs))]
        elif self.split == 'val':
            self.dh.triplets_df = self.dh.triplets_df[
                self.dh.triplets_df['song_id'].isin(val_songs)]
        elif self.split == 'test':
            self.dh.triplets_df = self.dh.triplets_df[
                ~self.dh.triplets_df['song_id'].isin(train_songs)]

    def _sample(self, X, length, dim=1):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        rand_start = np.random.randint(0, X.size()[dim] - length)
        if dim == 0:
            X = X[rand_start:rand_start + length]
        elif dim == 1:
            X = X[:, rand_start:rand_start + length]
        else:
            raise ValueError("dim must be 0 or 1.")
        return X

    def __len__(self):
        """Return length of the dataset."""
        return self.dh.triplets_df.shape[0]

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

        sample = {'u': user_idx, 'X': X, 'y': y}

        if self.transform:
            sample = self.transform(sample)

        return sample
