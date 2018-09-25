"""Datasets for training models in PyTorch."""

import numpy as np
import pandas as pd
import torch

from dc.datasets.dcuedataset import DCUEDataset


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
        self.triplets_df_user = None
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
        song_id = self.triplets_df_user.iloc[i]['song_id']
        song_idx = self.songid2metaindex[song_id]

        user_idx = self.dh.user_index[self.triplets_df_user.iloc[i]['user_id']]
        user_idx = torch.tensor(user_idx)

        score = self.triplets_df_user.iloc[i]['score']
        y = torch.from_numpy(np.array(score)).float()

        sample = {'u': user_idx, 'y': y, 'song_idx': song_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample
