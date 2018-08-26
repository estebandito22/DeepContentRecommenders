"""Datasets for training models in PyTorch."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dc.dcbr.cf.datahandler import CFDataHandler


class DCBRDataset(Dataset):

    """Class to load dataset required for training DCBR model."""

    def __init__(self, metadata_csv, factors_csv, split='train',
                 mode='dev', return_id=False, transform=None,
                 test_small=False, excluded_ids=None, data_type='scatter'):
        """
        Initialize MSDDataset.

        Args
            metadata_csv: Path to csv with the metadata for audio files to be
                          loaded.
            factors_csv: Path to csv containing the item factors learned from
                         collaborative filtering model.
            split: Which split of the data to load.  'train', 'val' or 'test'.
            mode: Running the model in development 'dev' or inference
                  'inference' mode.
            return_id: Boolean to return the id of the audio file in the
                       sample.
            transform: A method that transforms the sample.
            test_small: Boolean to use only the first 32 samples in the
                        metadata_csv.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        self.metadata_csv = metadata_csv
        self.factors_csv = factors_csv
        self.split = split
        self.mode = mode
        self.return_id = return_id
        self.transform = transform
        self.test_small = test_small
        self.excluded_ids = excluded_ids
        self.data_type = data_type

        # create metadata dataframe
        self.metadata = pd.read_csv(metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(self.excluded_ids)]

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

        # limit to split data and load target data if in dev mode
        if self.mode == 'dev':
            # self.metadata = self.metadata[self.metadata['split']
            # == self.split]
            self.target_data = np.loadtxt(factors_csv)

        # limit to small dataset
        if self.test_small:
            self.metadata = self.metadata.iloc[:32]

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        """Return sample idx from the dataset."""
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].iloc[idx])
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].iloc[idx])
            X = X.t()

        # only load target vectos in development mode,
        # otherwise use placeholder
        if self.mode == 'dev':
            y = self.target_data[self.metadata['item_index'].iloc[idx]]
        else:
            y = np.array([-1])

        sample = {'data': X, 'target': y}

        if self.transform:
            sample = self.transform(sample)

        # return the song_id with data when making predictions so save files
        # are identifiable
        if not self.return_id:
            return sample

        return sample, self.metadata['song_id'].iloc[idx]


class DCBRPredset(Dataset):

    """Class to load dataset required for predictions with DCBR model."""

    def __init__(self, metadata_csv, factors_csv, dh, split='train',
                 transform=None, excluded_ids=None, data_type='scatter'):
        """
        Initialize MSDDataset.

        Args
            metadata_csv: Path to csv with the metadata for audio files to be
                          loaded.
            factors_csv: Path to csv containing the user factors learned from
                         collaborative filtering model.
            item_user: item_user matrix to generate predictions on.
            split: Which split of the data to load.  'train', 'val' or 'test'.
            mode: Running the model in development 'dev' or inference
                  'inference' mode.
            return_id: Boolean to return the id of the audio file in the
                       sample.
            transform: A method that transforms the sample.
            test_small: Boolean to use only the first 32 samples in the
                        metadata_csv.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        self.metadata_csv = metadata_csv
        self.factors_csv = factors_csv
        self.dh = dh
        self.split = split
        self.transform = transform
        self.excluded_ids = excluded_ids
        self.data_type = data_type

        # create metadata dataframe
        self.metadata = pd.read_csv(metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(self.excluded_ids)]

        # create train and val and test splits
        if self.split == 'train':
            self.pred_item_user = dh.item_user_train
            self.filt_item_user = dh.item_user_test
        elif self.split == 'test':
            self.pred_item_user = dh.item_user_test
            self.filt_item_user = dh.item_user_train

        # create user and item lookups
        self.itemindex2songid = {v: k for (k, v) in dh.item_index.items()}

        # load user factors
        self.user_factors = np.loadtxt(factors_csv)

        self.metadata_user = self.metadata
        self.targets_user = None
        self.user_factor = None
        self.user_songids = None
        self.user_has_songs = False

    def create_user_data(self, i):
        """
        Create metadata file for all songs in prediction set less other songs.

        Args
            i: user index.
        """
        user_songs_idxs = self.pred_item_user.getcol(i).nonzero()[0]
        user_songs = [self.itemindex2songid[j] for j in user_songs_idxs]
        songs_mask = self.metadata['song_id'].isin(user_songs)
        self.user_songids = self.metadata['song_id'][songs_mask]

        if not self.user_songids.empty:
            self.user_has_songs = True
        else:
            self.user_has_songs = False

        user_filt_idxs = self.filt_item_user.getcol(i).nonzero()[0]
        user_filt = [self.itemindex2songid[j] for j in user_filt_idxs]
        filt_mask = ~self.metadata['song_id'].isin(user_filt)

        self.metadata_user = self.metadata[songs_mask | filt_mask]
        self.targets_user = np.where(self.metadata_user[
            'song_id'].isin(self.user_songids), 1.0, 0.0)

        self.user_factor = self.user_factors[i]

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata_user.shape[0]

    def __getitem__(self, i):
        """Return sample idx from the dataset."""
        if self.data_type == 'scatter':
            X = torch.load(self.metadata_user['data'].iloc[i])
        elif self.data_type == 'mel':
            X = torch.load(self.metadata_user['data_mel'].iloc[i])
            X = X.t()

        y = torch.tensor(self.targets_user.iloc[i])
        u = torch.from_numpy(self.user_factor)
        i = self.metadata_user.iloc[i].index.values

        sample = {'data': X, 'target': y, 'u': u, 'i': i}

        if self.transform:
            sample = self.transform(sample)

        return sample

class DCUEDataset(Dataset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets_txt, metadata_csv, neg_samples, split,
                 data_type, n_users=20000, n_items=10000, transform=None,
                 excluded_ids=None):
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
        self.excluded_ids = excluded_ids

        # load metadata and exclude ids
        self.metadata = pd.read_csv(metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(excluded_ids)]

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

        # remaining class attributes
        self.neg_samples = neg_samples
        self.data_type = data_type
        self.split = split

        # create train and val and test splits
        np.random.seed(10)
        train_mask = np.random.rand(self.taste_df.shape[0]) < 0.90
        np.random.seed(10)
        val_mask = np.random.rand(sum(train_mask)) < 0.2/0.9

        if self.split == 'train':
            self.dh.triplets_df = self.dh.triplets_df[train_mask]
            self.dh.triplets_df = self.dh.triplets_df[~val_mask]
        elif self.split == 'val':
            self.dh.triplets_df = self.dh.triplets_df[train_mask]
            self.dh.triplets_df = self.dh.triplets_df[val_mask]
        elif self.split == 'test':
            self.dh.triplets_df = self.dh.triplets_df[~train_mask]

    def _user_nonitem_songids(self, user_id):
        """
        Sample negative items for user.

        Args
            user_id: a user id from the triplets_txt file.
        """
        i = self.dh.user_index[user_id]
        items = self.dh.item_user.getcol(i).nonzero()[0]
        nonitems = self.all_items[~np.in1d(self.all_items, items)]
        sample_nonitems = np.random.choice(nonitems, self.neg_samples)
        return [self.itemindex2songid[idx] for idx in sample_nonitems]

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
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[pos_song_idx])

        # load torch negative tensor
        Ns = []
        for idx in neg_song_idxs:
            if self.data_type == 'scatter':
                N = torch.load(self.metadata['data'].loc[idx])
            elif self.data_type == 'mel':
                N = torch.load(self.metadata['data_mel'].loc[idx])
            Ns += [N]
        Ns = torch.stack(Ns)

        # returned for user embedding
        user_idx = self.user_index[self.dh.triplets_df.iloc[i]['user_id']]

        y = torch.tensor([-1])

        sample = {'u': user_idx, 'y': y, 'pos': X, 'neg': Ns}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DCUEPredset(Dataset):

    """Class to load data for predicting DCUE model."""

    def __init__(self, triplets_txt, metadata_csv, split, data_type,
                 n_users=20000, n_items=10000, transform=None,
                 excluded_ids=None):
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
        self.excluded_ids = excluded_ids

        # load metadata and exclude ids
        self.metadata = pd.read_csv(metadata_csv)
        if self.excluded_ids is not None:
            self.metadata = self.metadata[
                ~self.metadata['song_id'].isin(excluded_ids)]

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

        # remaining class attributes
        self.data_type = data_type
        self.split = split

        # Unique songs for creating user dataset
        self.taste_df_uniq_songs = self.taste_df['song_id'].unique()

        # create train and val and test splits
        np.random.seed(10)
        train_mask = np.random.rand(self.taste_df.shape[0]) < 0.90
        np.random.seed(10)
        val_mask = np.random.rand(sum(train_mask)) < 0.2/0.9

        if self.split == 'train':
            self.dh.triplets_df = self.dh.triplets_df[train_mask]
            self.dh.triplets_df = self.dh.triplets_df[~val_mask]
        elif self.split == 'val':
            self.dh.triplets_df = self.dh.triplets_df[train_mask]
            self.dh.triplets_df = self.dh.triplets_df[val_mask]
        elif self.split == 'test':
            self.dh.triplets_df = self.dh.triplets_df[~train_mask]

        # user data sets
        self.taste_df_user = self.taste_df
        self.non_taste_df_user = self.non_taste_df
        self.taste_df_user_comp = None
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
        return [self.itemindex2songid[idx] for idx in nonitems]

    def create_user_data(self, user_id):
        """
        Build a user specific dataset to predict from.

        Args
            user_id: A user id from the triplets_txt data.
        """
        self.taste_df_user = self.taste_df[
            self.taste_df['user_id'] == user_id]

        if not self.taste_df_user.empty:
            self.user_has_songs = True
        else:
            self.user_has_songs = False

        user_non_songs = self._user_nonitem_songids(user_id)

        self.taste_df_user_comp = pd.DataFrame(
            {'user_id': [user_id]*len(user_non_songs),
             'song_id': user_non_songs,
             'score': [0]*len(user_non_songs)})
        self.taste_df_user = pd.concat(
            [self.taste_df_user, self.taste_df_user_comp])

    def __len__(self):
        """Length of the user dataset."""
        return self.taste_df_user.shape[0]

    def __getitem__(self, i):
        """Return sample from the user dataset."""
        pos_song_id = self.taste_df_user.iloc[i]['song_id']
        pos_song_idx = self.song2index[pos_song_id]

        # load torch positive tensor
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].loc[pos_song_idx])
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[pos_song_idx])
            X = X.t()

        # returned for user embedding
        user_idx = self.user_index[self.taste_df_user.iloc[i]['user_id']]

        score = self.taste_df_user.iloc[i]['score']
        if score > 0:
            score = 1
        y = torch.from_numpy(np.array(score)).float()

        sample = {'u': user_idx, 'y': y, 'pos': X}

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


class RandomSample(object):

    """Take a random sample of a tensor along the first dimension."""

    def __init__(self, length, dim):
        """Initialize RandomSample."""
        self.length = length
        self.dim = dim

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
        X = sample['data']
        rand_start = np.random.randint(0, X.size()[self.dim] - self.length)

        if self.dim == 0:
            X = X[rand_start:rand_start + self.length]
        elif self.dim == 1:
            X = X[:, rand_start:rand_start + self.length]

        sample['data'] = X
        return sample


# if __name__ == '__main__':
#
#     from torch.utils.data import DataLoader
#
#     CUR_DIR = os.getcwd()
#     metadata_csv = os.path.join(CUR_DIR, "input", "MSD", "tracks.csv")
#     FACTORS_CSV = os.path.join(CUR_DIR, "recommender", "models",
#                                "fact_50_reg_0.01_iter_15_eps_1.csv")
#
#     train_data = MSDDataset(metadata_csv=metadata_csv,
#                             factors_csv=FACTORS_CSV,
#                             split='train',
#                             transform=ToTensor())
#
#     train_loader = DataLoader(train_data, batch_size=4,
#                               shuffle=True, num_workers=4)
#
#     for batch_idx, batch_samples in enumerate(train_loader):
#         print(batch_samples['data'].permute(1,0,2), batch_samples['target'])
#         if batch_idx>10:
#             break
