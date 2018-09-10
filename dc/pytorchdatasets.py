"""Datasets for training models in PyTorch."""

from collections import defaultdict

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

        # load metadata and exclude ids
        self.metadata = pd.read_csv(metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(excluded_ids)]
            self.metadata = self.metadata.reset_index(drop=True)

        # load taste profile dataset
        self.dh = CFDataHandler(triplets_txt)

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

        # lookup tables
        self.songid2metaindex = {v: k for (k, v)
                                 in self.metadata['song_id'].to_dict().items()}
        self.itemindex2songid = {v: k for (k, v)
                                 in self.dh.item_index.items()}

        # build all items to sample from
        self.n_items = self.dh.item_user.shape[0]
        self.n_users = self.dh.item_user.shape[1]
        self.all_items = np.arange(0, self.n_items)

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

        self.uniq_songs = self.dh.triplets_df['song_id'].unique()
        self.uniq_song_idxs = [self.dh.item_index[song_id] for
                               song_id in self.uniq_songs]

    def _user_nonitem_songids(self, user_id):
        """
        Sample negative items for user.

        Args
            user_id: a user id from the triplets_txt file.
        """
        i = self.dh.user_index[user_id]
        items = self.dh.item_user.getcol(i).nonzero()[0]
        nonitems = self.all_items[
            (~np.in1d(self.all_items, items)) &
            (np.in1d(self.all_items, self.uniq_song_idxs))]
        np.random.seed()
        if self.neg_samples is not None:
            nonitems = np.random.choice(nonitems, self.neg_samples)
        return [self.itemindex2songid[idx] for idx in nonitems]

    def _sample(self, X, length):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        rand_start = np.random.randint(0, X.size()[0] - length)
        X = X[rand_start:rand_start + length]
        return X

    def __len__(self):
        """Return length of the dataset."""
        return self.dh.triplets_df.shape[0]

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        user_id = self.dh.triplets_df.iloc[i]['user_id']
        pos_song_id = self.dh.triplets_df.iloc[i]['song_id']
        pos_song_idx = self.songid2metaindex[pos_song_id]

        # negative samples
        neg_song_ids = self._user_nonitem_songids(user_id)
        neg_song_idxs = [self.songid2metaindex[song_id]
                         for song_id in neg_song_ids]

        # load torch positive tensor
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].loc[pos_song_idx])
            X = self._sample(X, 17)
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[pos_song_idx])
            X = X.t()
            X = self._sample(X, 44)

        # load torch negative tensor
        Ns = []
        for idx in neg_song_idxs:
            if self.data_type == 'scatter':
                N = torch.load(self.metadata['data'].loc[idx])
                N = self._sample(N, 17)
            elif self.data_type == 'mel':
                N = torch.load(self.metadata['data_mel'].loc[idx])
                N = N.t()
                N = self._sample(N, 44)
            Ns += [N]
        Ns = torch.stack(Ns)

        # returned for user embedding
        user_idx = self.dh.user_index[self.dh.triplets_df.iloc[i]['user_id']]
        user_idx = torch.tensor(user_idx)

        y = torch.tensor([-1.0]).float()

        sample = {'u': user_idx, 'y': y, 'pos': X, 'neg': Ns}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DCUEPredset(DCUEDataset):

    """Class to load data for predicting DCUE model."""

    def __init__(self, triplets_txt, metadata_csv, split='train',
                 data_type='mel', n_users=20000, n_items=10000, transform=None,
                 excluded_ids=None, random_seed=None):
        """
        Initialize DCUE dataset for predictions.

        Args
            triplets_txt: path to text file with triplets of fields.
                user_id, song_id, score.
            metadata_csv: Path to csv with the metadata for audio files to be
                loaded.
            neg_samples: The number of negative samples to draw.
            split: 'train', 'val' or 'test'.
            data_type: 'scatter' or 'mel'.
            transform: A method that transforms the sample.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        DCUEDataset.__init__(
            self, triplets_txt, metadata_csv, split=split, data_type=data_type,
            n_users=n_users, n_items=n_items, transform=transform,
            excluded_ids=excluded_ids, random_seed=random_seed)

        # user data sets
        self.triplets_df_user = self.dh.triplets_df
        self.user_has_songs = False

    def _user_nonitem_songids(self, user_id):
        """
        Sample negative items for user.

        Args
            user_id: a user id from the triplets_txt file.
        """
        i = self.dh.user_index[user_id]
        items = self.dh.item_user.getcol(i).nonzero()[0]
        nonitems = self.all_items[~np.in1d(self.all_items, items)]
        nonitems = self.all_items[
            (~np.in1d(self.all_items, items)) &
            (np.in1d(self.all_items, self.uniq_song_idxs))]
        return [self.itemindex2songid[idx] for idx in nonitems]

    def create_user_data(self, user_id):
        """
        Build a user specific dataset to predict from.

        Args
            user_id: A user id from the triplets_txt data.
        """
        self.triplets_df_user = self.dh.triplets_df[
            self.dh.triplets_df['user_id'] == user_id].copy()

        if not self.triplets_df_user.empty:
            self.user_has_songs = True
            self.triplets_df_user['score'] = 1
        else:
            self.user_has_songs = False

        user_non_songs = self._user_nonitem_songids(user_id)

        triplets_df_user_comp = pd.DataFrame(
            {'user_id': [user_id]*len(user_non_songs),
             'song_id': user_non_songs,
             'score': [0]*len(user_non_songs)})
        self.triplets_df_user = pd.concat(
            [self.triplets_df_user, triplets_df_user_comp])

    def __len__(self):
        """Length of the user dataset."""
        return self.triplets_df_user.shape[0]

    def __getitem__(self, i):
        """Return sample from the user dataset."""
        pos_song_id = self.triplets_df_user.iloc[i]['song_id']
        pos_song_idx = self.songid2metaindex[pos_song_id]

        user_idx = self.dh.user_index[self.triplets_df_user.iloc[i]['user_id']]
        user_idx = torch.tensor(user_idx)

        score = self.triplets_df_user.iloc[i]['score']
        y = torch.from_numpy(np.array(score)).float()

        sample = {'u': user_idx, 'y': y, 'pos_song_idx': pos_song_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


# class DCUEPredset(DCUEDataset):
#
#     """Class to load data for predicting DCUE model."""
#
#     def __init__(self, triplets_txt, metadata_csv,
#                  split='train', data_type='mel', n_users=20000, n_items=10000,
#                  transform=None, excluded_ids=None):
#         """
#         Initialize DCUE dataset for predictions.
#
#         Args
#             triplets_txt: path to text file with triplets of fields.
#                 user_id, song_id, score.
#             metadata_csv: Path to csv with the metadata for audio files to be
#                 loaded.
#             neg_samples: The number of negative samples to draw.
#             split: 'train', 'val' or 'test'.
#             data_type: 'scatter' or 'mel'.
#             transform: A method that transforms the sample.
#             excluded_ids: List of audio file ids to exclude from dataset.
#         """
#         DCUEDataset.__init__(
#             self, triplets_txt, metadata_csv,
#             split=split, data_type=data_type,
#             n_users=n_users, n_items=n_items, transform=transform,
#             excluded_ids=excluded_ids)
#
#         # user data sets
#         self.triplets_df_user = None
#         self.triplets_df_user_comp = None
#         self.user_has_songs = False
#
#     def _user_nonitem_songids(self, user_id):
#         """
#         Sample negative items for user.
#
#         Args
#             user_id: a user id from the triplets_txt file.
#         """
#         i = self.dh.user_index[user_id]
#         items = self.dh.item_user.getcol(i).nonzero()[0]
#         nonitems = self.all_items[~np.in1d(self.all_items, items)]
#         nonitems = self.all_items[
#             (~np.in1d(self.all_items, items)) &
#             (np.in1d(self.all_items, self.uniq_song_idxs))]
#         return [self.itemindex2songid[idx] for idx in nonitems]
#
#     def create_user_data(self, user_id):
#         """
#         Build a user specific dataset to predict from.
#
#         Args
#             user_id: A user id from the triplets_txt data.
#         """
#         self.triplets_df_user = self.dh.triplets_df[
#             self.dh.triplets_df['user_id'] == user_id].copy()
#
#         if not self.triplets_df_user.empty:
#             self.user_has_songs = True
#             self.triplets_df_user['score'] = 1
#         else:
#             self.user_has_songs = False
#
#         user_non_songs = self._user_nonitem_songids(user_id)
#
#         self.triplets_df_user_comp = pd.DataFrame(
#             {'user_id': [user_id]*len(user_non_songs),
#              'song_id': user_non_songs,
#              'score': [0]*len(user_non_songs)})
#         self.triplets_df_user = pd.concat(
#             [self.triplets_df_user, self.triplets_df_user_comp])
#
#     def __len__(self):
#         """Length of the user dataset."""
#         return self.triplets_df_user.shape[0]
#
#     def __getitem__(self, i):
#         """Return sample from the user dataset."""
#         pos_song_id = self.triplets_df_user.iloc[i]['song_id']
#         pos_song_idx = self.songid2metaindex[pos_song_id]
#
#         # load torch positive tensor
#         if self.data_type == 'scatter':
#             X = torch.load(self.metadata['data'].loc[pos_song_idx])
#             X = self._sample(X, 17)
#         elif self.data_type == 'mel':
#             X = torch.load(self.metadata['data_mel'].loc[pos_song_idx])
#             data_mel = self.metadata['data_mel'].loc[pos_song_idx]
#             X = X.t()
#             X = self._sample(X, 44)
#
#         user_idx = self.dh.user_index[self.triplets_df_user.iloc[i]['user_id']]
#         if user_idx == 137:
#             print("Score Data User: {}".format(self.triplets_df_user.iloc[i]['user_id']))
#         user_idx = torch.tensor(user_idx)
#
#         score = self.triplets_df_user.iloc[i]['score']
#         y = torch.from_numpy(np.array(score)).float()
#
#         sample = {'pos': X, 'u': user_idx, 'y': y, 'pos_song_idx': pos_song_idx, 'data_mel': data_mel}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample


class DCUEItemset(DCUEDataset):

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
        DCUEDataset.__init__(
            self, triplets_txt, metadata_csv, data_type=data_type,
            n_users=n_users, n_items=n_items, transform=transform,
            excluded_ids=excluded_ids, random_seed=random_seed)

        # filter metadata to only top items
        self.metadata = self.metadata[
            self.metadata['song_id'].isin(list(self.dh.item_index.keys()))]

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        pos_song_id = self.metadata.iloc[i]['song_id']
        pos_song_idx = self.songid2metaindex[pos_song_id]

        # load torch positive tensor
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].loc[pos_song_idx])
            X = self._sample(X, 17)
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[pos_song_idx])
            X = X.t()
            X = self._sample(X, 44)

        sample = {'pos': X, 'metadata_index': pos_song_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


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

        if 'pos' in sample:
            key = 'pos'
            inputs = sample[key]
            inputs -= mean
            sample[key] = inputs

        if 'neg' in sample:
            key = 'neg'
            inputs = sample[key]
            inputs -= mean
            sample[key] = inputs

        return sample


class RandomSample(object):

    """Take a random sample of a tensor along the first dimension."""

    def __init__(self, length, dim, key):
        """Initialize RandomSample."""
        self.length = length
        self.dim = dim
        self.key = key

    def __call__(self, sample):
        """
        Randomly sample data along the first dimension of length len.

        Args
            sample: a dictionary with a key 'data' containing a tensor.
            len: the length of the sample to take from the data tensor.
            dim: dimension to sample on.  0 or 1.

        Return
            the original sample with newly randomly sampled tensor.
        """
        X = sample[self.key]
        rand_start = np.random.randint(0, X.size()[self.dim] - self.length)

        if self.dim == 0:
            X = X[rand_start:rand_start + self.length]
        elif self.dim == 1:
            X = X[:, rand_start:rand_start + self.length]

        sample[self.key] = X
        return sample
