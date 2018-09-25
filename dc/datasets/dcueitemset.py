"""Datasets for training models in PyTorch."""

import torch

from dc.datasets.dcuedataset import DCUEDataset


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
        song_id = self.metadata.iloc[i]['song_id']
        song_idx = self.songid2metaindex[song_id]

        # load torch positive tensor
        if self.data_type == 'scatter':
            X = torch.load(self.metadata['data'].loc[song_idx])
            X = self._sample(X, 17, 0)
        elif self.data_type == 'mel':
            X = torch.load(self.metadata['data_mel'].loc[song_idx])
            X = self._sample(X, 44, 1)

        sample = {'X': X, 'metadata_index': song_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample
