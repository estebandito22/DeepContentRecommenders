"""Classes to train Deep Content Recommender Models."""

import os
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from dc.pytorchdatasets import DCUEDataset
from dc.pytorchdatasets import DCUEPredset
from dc.pytorchdatasets import DCUEItemset
from dc.pytorchdatasets import SubtractMean

from dc.dcue.dcue import DCUENet

from dc.nn.trainer import Trainer


class DCUE(Trainer):

    """Class to train and evaluate DCUE model."""

    def __init__(self, feature_dim=100, batch_size=64, neg_batch_size=20,
                 u_embdim=300, margin=0.2, lr=0.00001, beta_one=0.9,
                 beta_two=0.99, eps=1e-8, weight_decay=0, num_epochs=100,
                 bn_momentum=0.5, model_type='mel', data_type='mel',
                 n_users=20000, n_items=10000, eval_pct=0.025):
        """
        Initialize DCUE model.

        Args
            feature_dim: Dimension of the feature vector embeddings.
            batch_size: Batch size to use for training.
            neg_batch_size: Number of negative samples to use per positive
                sample.
            u_embdim: Embedding dimension of the user lookup.
            margin: Hinge loss margin.
            lr: learning rate for ADAM optimizer.
            beta_one: Beta 1 parameter for ADAM optimizer.
            beta_two: Beta 2 parameter for ADAM optimizer.
            eps: EPS parameter for ADAM optimizer.
            weight_decay: Weight decay paramter for ADAM optimzer.
            bn_momentum: Momentum to use in batch normalization.
            num_epochs: Number of epochs to train.
            data_type: 'mel' or 'scatter'.
            n_users: number of users to include.
            n_items: number of items to include.
        """
        Trainer.__init__(self)

        # Trainer attributes
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.neg_batch_size = neg_batch_size
        self.u_embdim = u_embdim
        self.margin = margin
        self.lr = lr
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.eps = eps
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.bn_momentum = bn_momentum
        self.model_type = model_type
        self.data_type = data_type
        self.n_users = n_users
        self.n_items = n_items
        self.eval_pct = eval_pct

        # Dataset attributes
        self.model_dir = None
        self.triplets_txt = None
        self.metadata_csv = None
        self.save_dir = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pred_data = None
        self.item_data = None

        # Model attributes
        self.model = None
        self.optimizer = None
        self.loss_func = None
        self.dict_args = None
        self.nn_epoch = None

        self.item_factors = None
        self.user_factors = None

        self.best_item_factors = None
        self.best_user_factors = None
        self.best_auc = None
        self.best_val_loss = float('inf')

        if torch.cuda.is_available():
            self.USE_CUDA = True

    def load_dataset(self, split, transform=None, excluded_ids=None):
        """
        Load datasets for NN.

        Args
            split: the split of the data to load
            transform: transformation to use for NN dataset
            excluded_ids: ids to exclude from the metadata_csv for NN model.
        """
        if split == 'train':
            self.train_data = DCUEDataset(
                self.triplets_txt, self.metadata_csv, self.neg_batch_size,
                split, data_type=self.data_type, n_users=self.n_users,
                n_items=self.n_items, transform=transform,
                excluded_ids=excluded_ids)

        elif split == 'val':
            self.val_data = DCUEDataset(
                self.triplets_txt, self.metadata_csv, self.neg_batch_size,
                split, data_type=self.data_type, n_users=self.n_users,
                n_items=self.n_items, transform=transform,
                excluded_ids=excluded_ids, random_seed=10)

        elif split == 'test':
            self.test_data = DCUEDataset(
                self.triplets_txt, self.metadata_csv, self.neg_batch_size,
                split, data_type=self.data_type, n_users=self.n_users,
                n_items=self.n_items, transform=transform,
                excluded_ids=excluded_ids, random_seed=10)

    def load_pred_dataset(self, split, transform=None, excluded_ids=None):
        """
        Load data for predicting.

        Args
            split: the split of the data to load
            transform: transformation to use for NN dataset
            excluded_ids: ids to exclude from the metadata_csv for NN model.
        """
        self.pred_data = DCUEPredset(
            self.triplets_txt, self.metadata_csv, split,
            data_type=self.data_type, n_users=self.n_users,
            n_items=self.n_items, transform=transform,
            excluded_ids=excluded_ids, random_seed=10)

    def load_item_dataset(self, transform=None, excluded_ids=None):
        """
        Load data for generating item_factors.

        Args:
            transform: transformation to use for NN dataset
            excluded_ids: ids to exclude from the metadata_csv for NN model.
        """
        self.item_data = DCUEItemset(
            self.triplets_txt, self.metadata_csv, data_type=self.data_type,
            n_users=self.n_users, n_items=self.n_items, transform=transform,
            excluded_ids=excluded_ids, random_seed=10)

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.dict_args = {'data_type': self.data_type,
                          'feature_dim': self.feature_dim,
                          'user_embdim': self.u_embdim,
                          'user_count': self.train_data.n_users,
                          'bn_momentum': self.bn_momentum,
                          'model_type': self.model_type}

        self.model = DCUENet(self.dict_args)

        self.loss_func = nn.HingeEmbeddingLoss(margin=self.margin)
        self.optimizer = optim.Adam(
            self.model.parameters(), self.lr,
            (self.beta_one, self.beta_two),
            self.eps, self.weight_decay)

        if self.USE_CUDA:
            self.model = self.model.cuda()
            self.loss_func = self.loss_func.cuda()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        losses_processed = 0

        for batch_samples in loader:

            # prepare training sample
            u = batch_samples['u']
            y = batch_samples['y']
            # batch size x seqdim x seqlen
            pos = batch_samples['X']
            # batch size x neg batch size x seqdim x seqlen
            neg = self._withinbatch_negsample(pos)

            if self.model_type.find('2d') > -1:
                # batch size x 1 x seqdim x seqlen
                pos = pos.unsqueeze(1)
                # batch size x neg batch size x 1 x seqdim x seqlen
                neg = neg.unsqueeze(2)

            if self.USE_CUDA:
                u = u.cuda()
                y = y.cuda()
                pos = pos.cuda()
                neg = neg.cuda()

            # forward pass
            self.model.zero_grad()
            preds = self.model(u, pos, neg)

            # backward pass
            loss = self.loss_func(preds, y)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += pos.size()[0]
            losses_processed += preds.numel()

            train_loss += loss.item() * preds.numel()

        train_loss /= losses_processed

        return samples_processed, train_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        losses_processed = 0

        with torch.no_grad():
            for batch_samples in loader:

                # prepare training sample
                u = batch_samples['u']
                y = batch_samples['y']
                # batch size x seqdim x seqlen
                pos = batch_samples['X']
                # batch size x neg batch size x seqdim x seqlen
                neg = self._withinbatch_negsample(pos)

                if self.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    pos = pos.unsqueeze(1)
                    # batch size x neg batch size x 1 x seqdim x seqlen
                    neg = neg.unsqueeze(2)

                if self.USE_CUDA:
                    u = u.cuda()
                    y = y.cuda()
                    pos = pos.cuda()
                    neg = neg.cuda()

                # forward pass
                preds = self.model(u, pos, neg)

                # compute loss
                loss = self.loss_func(preds, y)

                samples_processed += pos.size()[0]
                losses_processed += preds.numel()

                val_loss += loss.item() * preds.numel()

            val_loss /= losses_processed

        return samples_processed, val_loss

    def fit(self, triplets_txt, metadata_csv, save_dir):
        """
        Train the NN model.

        Args
            triplets_txt: path to the triplets_txt file.
            metadata_csv: path to the metadata_csv file.
            save_dir: directory to save nn_model
        """
        # Print settings to output file
        print("Feature Dim: {}\n\
               Batch Size: {}\n\
               Negative Batch Size: {}\n\
               User Embedding Dim: {}\n\
               Margin: {}\n\
               Learning Rate: {}\n\
               Beta One: {}\n\
               Beta Two: {}\n\
               EPS: {}\n\
               Weight Decay: {}\n\
               Num Epochs: {}\n\
               Model Type: {}\n\
               Data Type: {}\n\
               Num Users: {}\n\
               Num Items: {}\n\
               Save Dir: {}".format(
                   self.feature_dim, self.batch_size, self.neg_batch_size,
                   self.u_embdim, self.margin, self.lr, self.beta_one,
                   self.beta_two, self.eps, self.weight_decay, self.num_epochs,
                   self.model_type, self.data_type, self.n_users, self.n_items,
                   save_dir))

        self.model_dir = save_dir
        self.metadata_csv = metadata_csv
        self.triplets_txt = triplets_txt

        print("Loading datasets...")
        self.load_dataset(
            split='train', transform=SubtractMean(self.data_type))
        self.load_dataset(
            split='val', transform=SubtractMean(self.data_type))
        self.load_item_dataset(transform=SubtractMean(self.data_type))
        self.load_pred_dataset(
            split='val', transform=SubtractMean(self.data_type))

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)
        pred_loader = DataLoader(
            self.pred_data, batch_size=1024, shuffle=True, num_workers=4)

        self._init_nn()

        # init training variables
        train_loss = 0
        samples_processed = 0
        self.best_auc = 0
        self.best_val_loss = float('inf')

        # train loop
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch
            if epoch > 0:
                print("Initializing train epoch...")
                start = datetime.datetime.now()
                sp, train_loss = self._train_epoch(train_loader)
                samples_processed += sp
                end = datetime.datetime.now()
                print("Train epoch processed in {}".format(end-start))

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss:{}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss))

            if epoch % 5 == 0:
                # compute loss
                print("Initializing val epoch...")
                start = datetime.datetime.now()
                _, val_loss = self._eval_epoch(val_loader)
                end = datetime.datetime.now()
                print("Val epoch processed in {}".format(end-start))

                # compute auc estimate.  Gives +/- approx 0.017 @ 95%
                # confidence w/ 20K users.
                print("Initializing AUC computation...")
                start = datetime.datetime.now()
                val_auc = self._compute_auc(
                    'val', pred_loader, pct=self.eval_pct)
                end = datetime.datetime.now()
                print("AUC computation processed in {}".format(end-start))

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: \
                {}\tValidation Loss: {}\tValidation AUC: {}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    val_loss, val_auc))

            self._update_best(val_auc, val_loss)

    def score(self, users, loader, k=10000):
        """
        Score the model with AUC.

        Args
            users: list of user_ids
            split: split to score on
        """
        self.model.eval()

        auc = []
        for user_id in users:
            scores, targets = self.predict(user_id, loader)

            if (scores is not None) and (targets is not None):
                scores, targets = list(zip(
                    *sorted(zip(scores, targets), key=lambda x: -x[0])))

                scores = scores[:k]
                targets = targets[:k]

                if sum(targets) == len(targets):
                    auc += [1]
                elif sum(targets) == 0:
                    auc += [0]
                else:
                    auc += [roc_auc_score(targets, scores)]

        return np.mean(auc)

    def predict(self, user, loader):
        """
        Predict for a user.

        Args
            user: a user id
        """
        loader.dataset.create_user_data(user)
        self.model.eval()

        if loader.dataset.user_has_songs:

            with torch.no_grad():
                scores = []
                targets = []
                for batch_samples in loader:

                    u = []
                    for idx in batch_samples['u']:
                        u += [self.user_factors[idx]]
                    u = torch.stack(u)

                    i = []
                    for idx in batch_samples['song_idx']:
                        i += [self.item_factors[idx]]
                    i = torch.stack(i)

                    y = batch_samples['y']

                    if i.size()[0] > 1:
                        if self.USE_CUDA:
                            u = u.cuda()
                            i = i.cuda()

                        # forward pass
                        score = self.model.sim(u, i)
                        scores += score.cpu().numpy().tolist()
                        targets += y.numpy().tolist()

            return scores, targets

        return None, None

    def insert_best_factors(self):
        """Insert the best factors for predictions."""
        self.item_factors = self.best_item_factors
        self.user_factors = self.best_user_factors

    def _update_best(self, val_auc, val_loss):

        if val_auc > self.best_auc:
            self.best_auc = val_auc

            if self.best_item_factors is None or \
               self.best_user_factors is None:
                self.best_item_factors = torch.zeros_like(
                    self.item_factors)
                self.best_user_factors = torch.zeros_like(
                    self.user_factors)

            self.best_item_factors.copy_(self.item_factors)
            self.best_user_factors.copy_(self.user_factors)

            self.save(models_dir=self.model_dir)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

    def _compute_auc(self, split, loader, pct=0.025):
        self._user_factors()
        self._item_factors()

        if split == 'train':
            users = list(self.train_data.dh.user_index.keys())
        elif split == 'val':
            users = list(self.val_data.dh.user_index.keys())
        elif split == 'test':
            users = list(self.test_data.dh.user_index.keys())

        n_users = len(users)
        if pct < 1:
            users_sample = np.random.choice(users, int(n_users * pct))
        else:
            users_sample = users

        return self.score(users_sample, loader)

    def _user_factors(self):
        """Create user factors matrix."""
        self.user_factors = torch.zeros([self.n_users, self.feature_dim])
        self.model.eval()
        with torch.no_grad():
            for i in list(self.item_data.dh.user_index.values()):
                emb_idx = torch.tensor([i])
                if self.USE_CUDA:
                    emb_idx = emb_idx.cuda()
                self.user_factors[i] = self.model.user_embd(emb_idx)

    def _item_factors(self):
        """Create item factors matrix."""
        item_loader = DataLoader(
            self.item_data, batch_size=self.batch_size, shuffle=False,
            num_workers=4)

        self.item_factors = torch.zeros(
            [len(self.item_data.songid2metaindex), self.feature_dim])

        self.model.eval()
        with torch.no_grad():
            for batch_samples in item_loader:
                # batch size x seqdim x seqlen
                X = batch_samples['X']
                if self.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    X = X.unsqueeze(1)
                metadata_indexes = batch_samples['metadata_index']

                if self.USE_CUDA:
                    X = X.cuda()

                item_factor = self.model.conv(X)

                for i, idx in enumerate(metadata_indexes):
                    self.item_factors[idx] = item_factor[i]

    def _withinbatch_negsample(self, song_batch):
        batch_size, seqdim, seqlen = song_batch.size()
        neg = torch.zeros([batch_size, self.neg_batch_size, seqdim, seqlen])

        for i in range(batch_size):
            indexes = [x for x in range(0, i)] + \
                      [x for x in range(i+1, batch_size)]
            for j in range(self.neg_batch_size):
                rand_idx = np.random.choice(indexes)
                neg[i][j].copy_(song_batch[rand_idx])

        return neg

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = "DCUE_lr_{}_b1_{}_b2_{}_wd_{}_nu_{}_ni_{}_mt_{}".\
                format(self.lr, self.beta_one, self.beta_two,
                       self.weight_decay, self.n_users, self.n_items,
                       self.model_type)

            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained DCUE model.

        Args
            model_dir: directory where models are saved.
            epoch: epoch of model to load.
        """
        epoch_file = "epoch_"+str(epoch)+".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])
