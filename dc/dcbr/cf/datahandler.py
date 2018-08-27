"""Class to perform data manipulations necessary when training CF model."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from dc.dcbr.cf.evaluation import train_test_split as tts


class CFDataHandler():

    """Manipulate data necessary for training CF Model."""

    def __init__(self, triplets_txt):
        """
        Initialize data handler.

        Args
            triplets_txt: path to text file with triplets of fields.
                user_id, song_id, score.
        """
        colnames = ['user_id', 'song_id', 'score']
        self.triplets_df = pd.read_table(
            triplets_txt, header=None, names=colnames)

        self.item_user = None
        self.user_index = None
        self.item_index = None
        # self.item_user_matrix()

        self.item_user_train = None
        self.item_user_test = None

    def item_user_matrix(self):
        """
        Build the csr matrix of item x user.

        Args
            X: Dataframe with 'item', 'user', 'score' fields.

        Return
            item_user: csr matrix with items as rows, users as columns and
                score as values.
            item_index: dictionary of the item indexes in the item_user matrix
                for each item id.
            user_index: dicationary of the user indexes in the item_user matrix
                for each user_id.

        ---------------------------------------------------------------------------
        """
        self.triplets_df['user_id'] = self.triplets_df[
            'user_id'].astype("category")
        self.triplets_df['song_id'] = self.triplets_df[
            'song_id'].astype("category")

        row = self.triplets_df['song_id'].cat.codes.copy()
        col = self.triplets_df['user_id'].cat.codes.copy()

        nrow = len(self.triplets_df['song_id'].cat.categories)
        ncol = len(self.triplets_df['user_id'].cat.categories)

        self.item_user = csr_matrix((self.triplets_df['score'],
                                     (row, col)), shape=(nrow, ncol))

        user = dict(enumerate(self.triplets_df['user_id'].cat.categories))
        self.user_index = {u: i for i, u in user.items()}

        item = dict(enumerate(self.triplets_df['song_id'].cat.categories))
        self.item_index = {s: i for i, s in item.items()}

    def user_min_items(self, min_items):
        """
        Remove users from training data with less than min_items iteractions.

        Args
            min_items: The minimum number of items a user must have interacted
                with to remain in the dataset

        Return
            Dataframe excluding users that do not meet min_items requirement.

        ---------------------------------------------------------------------------
        """
        user_item_counts = self.triplets_df.groupby(
            'user_id')['song_id'].count()
        user_item_counts = user_item_counts[user_item_counts >= min_items]
        mask = np.where(
            self.triplets_df['user_id'].isin(user_item_counts.index))[0]
        self.triplets_df = self.triplets_df.loc[mask]

    def train_test_split(self, percentage):
        """
        Create a train-test split of the item_user matrix.

        Args
            percentage: Percent of data to use for training.
        """
        np.random.seed(10)
        self.item_user_train, self.item_user_test = tts(
            self.item_user, percentage)

    def train_val_split(self, percentage):
        """
        Create a train-validation split of the item_user matrix.

        Args
            percentage: Percent of data to use for training.
        """
        return tts(self.item_user_train, percentage)

    def sample_users(self, item_user_train=None, item_user_test=None,
                     percentage=0.25):
        """
        Sample users from the user x item matrix.

        To be used during evaluation to reduce computation time.

        Args
            percentage: percentage of users to sample.

        Return
            sample_train: csr matrix with users as rows, items as columns and
                          score as values, with data for sampled users
            sample_test: csr matrix with users as rows, items as columns and
                         score as values, with data for sampled users

        ---------------------------------------------------------------------------
        """
        if (item_user_train is None) and (item_user_test is None):
            user_item_train = self.item_user_train.transpose(copy=True).tocsr()
            user_item_test = self.item_user_test.transpose(copy=True).tocsr()
        elif (item_user_train is not None) and (item_user_test is not None):
            user_item_train = item_user_train.transpose(copy=True).tocsr()
            user_item_test = item_user_test.transpose(copy=True).tocsr()
        else:
            raise Exception("Must supply both item_user_train and \
             item_user_test or neither.")
        user_item_train = user_item_train.tocoo()
        user_item_test = user_item_test.tocoo()

        # create a random index and use as a mask to build a list of rows
        # (user) to use in the sample
        random_index = np.random.random(user_item_train.shape[0])
        sample_rows = np.arange(
            user_item_train.shape[0])[(random_index < percentage)]

        # build a boolean mask of the coo row attribute that can be used to
        # select
        # the rows columns and data that correspond to the randomly selected
        # rows
        sample_index_train = np.in1d(user_item_train.row, sample_rows)
        sample_index_test = np.in1d(user_item_test.row, sample_rows)

        # create new csr matricies with only the data from the sampled users
        # but with the same shape as the original matricies
        sample_train = csr_matrix((user_item_train.data[sample_index_train],
                                   (user_item_train.row[sample_index_train],
                                    user_item_train.col[sample_index_train])),
                                  shape=user_item_train.shape,
                                  dtype=user_item_train.dtype)

        sample_test = csr_matrix((user_item_test.data[sample_index_test],
                                  (user_item_test.row[sample_index_test],
                                   user_item_test.col[sample_index_test])),
                                 shape=user_item_test.shape,
                                 dtype=user_item_test.dtype)

        return sample_train, sample_test

    def item_complement(self, item_indexes):
        """
        Find the complement set of the given item indexes in item_user.

        Args
            item_indexes: the indexes in item_user to exclude.
        """
        n_items = self.item_user.shape[0]

        all_item_indexes = np.arange(0, n_items)
        item_indexes_complement = all_item_indexes[
            ~np.in1d(all_item_indexes, item_indexes)]

        return item_indexes_complement.tolist()
