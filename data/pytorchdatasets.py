"""Datasets for training models in PyTorch."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import item_user_matrix


class DCBRDataset(Dataset):

    """Class to load dataset required for training DCBR model."""

    def __init__(self, metadata_csv, factors_csv, split='train',
                 mode='dev', return_id=False, transform=None,
                 test_small=False, excluded_ids=None):
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

        # create metadata dataframe
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[
            ~self.metadata['song_id'].isin(self.excluded_ids)]

        # limit to split data and load target data if in dev mode
        if self.mode == 'dev':
            self.metadata = self.metadata[self.metadata['split'] == self.split]
            self.target_data = np.loadtxt(factors_csv)

        # limit to small dataset
        if self.test_small:
            self.metadata = self.metadata.iloc[:32]

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        """Return sample idx from the dataset."""
        X = torch.load(self.metadata['data'].iloc[idx])

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


class DCUEDataset(Dataset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets_txt, metadata_csv, neg_samples, input_type,
                 split, excluded_ids):
        """
        Initialize DCUE dataset.

        Args
            triplets_txt: path to text file with triplets of fields.
                user_id, song_id, score.
            metadata_csv: Path to csv with the metadata for audio files to be
                loaded.
            neg_samples: The number of negative samples to draw.
            input_type: 'scatter' or 'mel'.
            split: 'train' or 'val'.
            excluded_ids: List of audio file ids to exclude from dataset.
        """
        # load metadata and exclude ids
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[
            ~self.metadata['song_id'].isin(excluded_ids)]

        # load taste profile dataset
        colnames = ['user_id', 'song_id', 'score']
        self.taste_df = pd.read_table(
            triplets_txt, header=None, names=colnames)

        # limit taste df to songs with audio only
        self.taste_df = self.taste_df[
            self.taste_df['song_id'].isin(self.metadata['song_id'])]

        # limit taste df to most frequent songs and users
        top_items = self.taste_df.groupby(
            'song_id').count().nlargest(10000, 'user_id').index.values
        top_users = self.taste_df.groupby(
            'user_id').count().nlargest(20000, 'song_id').index.values
        top_items_mask = self.taste_df['song_id'].isin(top_items)
        top_users_mask = self.taste_df['user_id'].isin(top_users)
        self.taste_df = self.taste_df[(top_items_mask) & (top_users_mask)]

        # build item_user sparse matrix for quick item lookups
        self.item_user, self.item_index, self.user_index = item_user_matrix(
            self.taste_df)

        # lookup tables
        self.songid2metaindex = {v: k for (k, v)
                                 in self.metadata['song_id'].to_dict().items()}
        self.itemindex2songid = {v: k for (k, v)
                                 in self.item_index.items()}

        # build all items to sample from
        self.n_items = self.item_user.shape[0]
        self.n_users = self.item_user.shape[1]
        self.all_items = np.arange(0, self.n_items)

        # remaining class attributes
        self.neg_samples = neg_samples
        self.input_type = input_type
        self.split = split

        # create train and val and test splits
        np.random.seed(10)
        train_mask = np.random.rand(self.taste_df.shape[0]) < 0.70
        if self.split == 'train':
            self.taste_df = self.taste_df[train_mask]
        else:
            self.taste_df = self.taste_df[~train_mask]
            np.random.seed(10)
            val_mask = np.random.rand(self.taste_df.shape[0]) < 0.66
            if self.split == 'val':
                self.taste_df = self.taste_df[val_mask]
            else:
                self.taste_df = self.taste_df[~val_mask]

    def _user_nonitem_songids(self, user_id):
        """
        Sample negative items for user.

        Args
            user_id: a user id from the triplets_txt file.
        """
        i = self.user_index[user_id]
        items = self.item_user.getcol(i).nonzero()[0]
        nonitems = self.all_items[~np.in1d(self.all_items, items)]
        nonitems = nonitems[np.in1d(nonitems, self.metadata['item_index'])]
        sample_nonitems = np.random.choice(nonitems, self.neg_samples)
        return [self.itemindex2songid[idx] for idx in sample_nonitems]

    def __len__(self):
        """Return length of the dataset."""
        return self.taste_df.shape[0]

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        user_id = self.taste_df.iloc[i]['user_id']
        pos_song_id = self.taste_df.iloc[i]['song_id']
        pos_song_idx = self.songid2metaindex[pos_song_id]

        # negative samples
        neg_song_ids = self._user_nonitem_songids(user_id)
        neg_song_idxs = [self.songid2metaindex[song_id]
                         for song_id in neg_song_ids]

        # load torch positive tensor
        if self.input_type == 'scatter':
            X = torch.load(self.metadata['data'].loc[pos_song_idx])
        elif self.input_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[pos_song_idx])

        # load torch negative tensor
        Ns = []
        for idx in neg_song_idxs:
            if self.input_type == 'scatter':
                N = torch.load(self.metadata['data'].loc[idx])
            elif self.input_type == 'mel':
                N = torch.load(self.metadata['data_mel'].loc[idx])
            Ns += [N]
        Ns = torch.stack(Ns)

        # returned for user embedding
        user_idx = self.user_index[self.taste_df.iloc[i]['user_id']]

        y = torch.tensor([-1])

        sample = {'u': user_idx, 'y': y, 'pos': X, 'neg': Ns}

        return sample


class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """Return a pytorch tensor sample."""
        song, score = sample['data'], sample['target']

        return {'data': torch.from_numpy(song).float(),
                'target': torch.from_numpy(score).float()}


class RandomSample(object):

    """Take a random sample of a tensor along the first dimension."""

    def __call__(self, sample, length, dim):
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
        rand_start = np.random.randint(0, X.size()[dim] - length)

        if dim == 0:
            X = X[rand_start:rand_start + length]
        elif dim == 1:
            X = X[:, rand_start:rand_start + length]

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
