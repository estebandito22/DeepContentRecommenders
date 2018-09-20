"""Class for building 'playlists' based on layer activations."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd

from analysis.analyzer import BaseAnalyzer


class PlaylistAnalyzer(BaseAnalyzer):

    """Class to analyze models using playlist generation strategy."""

    def __init__(self, trainer, save_dir):
        """Initialize playlist analyzer."""
        BaseAnalyzer.__init__(self)
        self.trainer = trainer
        self.save_dir = save_dir
        self.item_activations = None
        self.model_name = None

    def load(self, model_dir, epoch):
        """Use model load function."""
        self.trainer.load(model_dir, epoch)
        self.model_name = model_dir.split("/")[-1] + "_epoch_" + str(epoch)

    def save(self, df):
        """Save."""
        file_path = os.path.join(self.save_dir, self.model_name + ".csv")
        df.to_csv(file_path, index=False)

    def _get_activations(self, layer_number):
        """
        Get activations for each input from layer numbered layer_number.

        Args
            layer_number: the number, indexed from 0, of the layer to analyze.
        """
        item_loader = DataLoader(
            self.trainer.item_data, batch_size=self.trainer.batch_size,
            shuffle=False, num_workers=4)

        first = True
        self.trainer.model.eval()
        with torch.no_grad():
            for batch_samples in item_loader:
                # batch size x seqdim x seqlen
                X = batch_samples['X']
                if self.trainer.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    X = X.unsqueeze(1)
                metadata_indexes = batch_samples['metadata_index']

                if self.trainer.USE_CUDA:
                    X = X.cuda()

                layer_model = nn.Sequential(
                    *list(self.trainer.model.conv.children())[:layer_number])
                output = layer_model(X)

                if first:
                    self.item_activations = torch.zeros(
                        [len(self.trainer.item_data.songid2metaindex),
                         output[0].view(1, -1).size()[1]])

                for i, idx in enumerate(metadata_indexes):
                    self.item_activations[idx] = output[i].view(1, -1)

    def analyze(self, layer_number):
        """
        Build a dataframe of activations and save.

        Args
            layer_number: the number, indexed from 0, of the layer to analyze.
        """
        self._get_activations(layer_number)

        item_activations_df = pd.DataFrame(
            index=self.trainer.item_data.songid2metaindex.values(),
            columns=['w{}'.format(i) for i
                     in range(self.item_activations.size()[1])],
            data=self.item_activations.numpy())
        meta_item_activations_df = self.trainer.item_data.metadata.join(
            item_activations_df)

        self.save(meta_item_activations_df)
