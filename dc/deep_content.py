"""Classes to train Deep Content Recommender Models."""

from abc import ABC, abstractmethod

import os
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from torchvision.transforms import Compose

from dc.dcbr.cf.wrmf import WRMF
from dc.dcbr.cf.datahandler import CFDataHandler
from dc.dcbr.cf.evaluators import cross_validate

from dc.pytorchdatasets import DCBRDataset
from dc.pytorchdatasets import DCBRPredset
from dc.pytorchdatasets import DCUEDataset
from dc.pytorchdatasets import DCUEPredset
from dc.pytorchdatasets import DCUEItemset

from dc.pytorchdatasets import ToTensor

from dc.dcbr.dcbr import DCBRNet
from dc.dcue.dcue import DCUENet


class Trainer(ABC):

    """Abstract class for a model trainer."""

    @abstractmethod
    def __init__(self):
        """Initialize abstract trainer class."""

    @abstractmethod
    def fit(self):
        """Train model."""
        raise NotImplementedError("train is an abstract method.")

    @abstractmethod
    def predict(self):
        """Evaluate model."""
        raise NotImplementedError("evaluate is an abstract method.")

    @abstractmethod
    def score(self):
        """Score model."""
        raise NotImplementedError("score is an abstract method.")

    @abstractmethod
    def save(self):
        """Save model."""
        raise NotImplementedError("save is an abstract method.")


class DCBR(Trainer):

    """Class to train DCBR model."""

    def __init__(self, factors=400, l2=1, alpha=1, cf_eps=1e-8, n_iter=15,
                 n_splits=5, train_pct=0.7, n_recs=500, eval_pct=0.01,
                 output_size=400, dropout=0, batch_size=32, lr=0.001,
                 beta_one=0.9, beta_two=0.999, nn_eps=1e-8, weight_decay=0,
                 num_epochs=100, bn_momentum=0.5, data_type='mel'):
        """
        Initialize the DCBR trainer.

        Args
            factors: The number of factors to use for the user and item
                embeddings.
            l2: [0,inf) the L2 regularization parameter.
            alpha: alpha paramter for score transformation.
            cf_eps: eps paramter for score transformation.
            n_iter: the number of iterations to use for ALS optimization.
            train_pct: the percentage of data to use for training.  The
                remaining data is used for validation.
            n_splits: The number of splits to use for cross validation.
            train_pct: The percentage of data to use in the training fold.
            n_recs: The number of recommendations to use for evaluation.
            eval_pct: The percent of users to use for evaluation.

            output_size: size of the target factor dim.
            dropout: dropout to use.
            batch_size: batch_size to use.
            lr: learning rate to use.
            beta_one: beta one parameter in ADAM.
            beta_two: beta two parameter in ADAM.
            nn_eps: eps parameter in ADAM.
            weight_decay: weight decay parameter in ADAM.
            bn_momentum: Momentum to use in batch normalization.
            num_epochs: number of epochs to train.
            data_type: 'mel' or 'scatter'
        """
        Trainer.__init__(self)

        # cf model attributes
        self.factors = factors
        self.l2 = l2
        self.alpha = alpha
        self.cf_eps = cf_eps
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.n_recs = n_recs
        self.eval_pct = eval_pct

        # nn model attributes
        self.output_size = output_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.nn_eps = nn_eps
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.data_type = data_type
        self.bn_momentum = bn_momentum

        # additional cf attributes
        self.triplets_txt = None
        self.dh = None
        self.wrmf = None

        # additional nn attributes
        self.metadata_csv = None
        self.models_dir = None
        self.wrmf_name = None
        self.u_factors_csv = None
        self.nn_train_data = None
        self.nn_val_data = None
        self.nn_test_data = None
        self.nn_pred_data = None
        self.nn_all_data = None
        self.nn_model = None
        self.nn_optimizer = None
        self.nn_loss_func = None
        self.nn_epoch = None
        self.nn_dict_args = None

        self.best_item_factors = None

        if torch.cuda.is_available():
            self.USE_CUDA = True

    def load_cf_dataset(self):
        """Load datasets for CF models."""
        # cf data
        if self.dh is None:
            self.dh = CFDataHandler(self.triplets_txt)
            self.dh.item_user_matrix()
            self.dh.train_test_split(0.9)

    def load_nn_dataset(self, split, transform=None, excluded_ids=None):
        """
        Load datasets for both NN.

        Args
            split: the split of the data to load
            transform: transformation to use for NN dataset
            excluded_ids: ids to exclude from the metadata_csv for NN model.
        """
        if split == 'train':
            if self.wrmf is not None:
                # nn data
                self.nn_train_data = DCBRDataset(
                    self.metadata_csv, self.dh.item_index, split,
                    transform=transform, excluded_ids=excluded_ids,
                    data_type=self.data_type)

        elif split == 'val':
            if self.wrmf is not None:
                # nn data
                self.nn_val_data = DCBRDataset(
                    self.metadata_csv, self.dh.item_index, split,
                    transform=transform, excluded_ids=excluded_ids,
                    data_type=self.data_type)

        elif split == 'test':
            if self.wrmf is not None:
                # nn data
                self.nn_test_data = DCBRDataset(
                    self.metadata_csv, self.dh.item_index, split,
                    transform=transform, excluded_ids=excluded_ids,
                    data_type=self.data_type)

        elif split == 'all':
            if self.wrmf is not None:
                # nn data
                self.nn_all_data = DCBRDataset(
                    self.metadata_csv, self.dh.item_index, split,
                    transform=transform, excluded_ids=excluded_ids,
                    data_type=self.data_type)

    def load_pred_dataset(self, split, transform=None, excluded_ids=None):
        """Load data for predicting."""
        if self.wrmf is not None:
            self.nn_pred_data = DCBRPredset(
                self.metadata_csv, self.dh, split=split, transform=transform,
                excluded_ids=excluded_ids)

    def fit_cf(self, triplets_txt):
        """
        Train CF model.

        Args
            triplets_txt: path to the triplets_txt file.
            save_dir: directory to save model.
        """
        self.triplets_txt = triplets_txt

        self.load_cf_dataset()

        self.wrmf = WRMF(self.factors, self.l2, self.alpha, self.cf_eps,
                         self.n_iter)

        if self.n_splits > 0:
            summary = cross_validate(
                self.dh, self.wrmf, self.n_splits, self.train_pct, self.n_recs,
                self.eval_pct)
            print(summary)
        else:
            raise Exception("n_splits must be > 0!")

        self.wrmf.fit(self.dh.item_user_train)
        self.best_item_factors = self.wrmf.wrmf.item_factors.copy()

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.nn_dict_args = {
            'output_size': self.output_size,
            'dropout': self.dropout,
            'init_fc_bias': self.wrmf.wrmf.item_factors.mean(0),
            'bn_momentum': self.bn_momentum,
            'data_type': self.data_type}

        self.nn_model = DCBRNet(self.nn_dict_args)

        self.nn_loss_func = nn.MSELoss()
        self.nn_optimizer = optim.Adam(
            self.nn_model.parameters(), self.lr,
            (self.beta_one, self.beta_two),
            self.nn_eps, self.weight_decay)

        if self.USE_CUDA:
            self.nn_model = self.nn_model.cuda()
            self.nn_loss_func = self.nn_loss_func.cuda()

    def _nn_train_epoch(self, loader):
        """Train epoch."""
        self.nn_model.train()
        train_loss = 0
        samples_processed = 0
        for batch_samples in loader:

            # batch size x 1 x scatter dim x seqlen
            inputs = batch_samples['data'].permute(0, 2, 1).unsqueeze(1)
            # batch size x 1
            target_index = batch_samples['target_index']
            # batch size x feature_dim
            target = []
            for idx in target_index:
                target += [torch.from_numpy(
                    self.wrmf.wrmf.item_factors[idx])]
            target = torch.stack(target)

            if self.USE_CUDA:
                inputs = inputs.cuda()
                target = target.cuda()

            # forward pass
            self.nn_model.zero_grad()
            pred = self.nn_model(inputs)

            # backward pass
            loss = self.nn_loss_func(pred, target)
            loss.backward()
            self.nn_optimizer.step()

            # compute train loss
            batch_size = inputs.size()[0]
            samples_processed += batch_size

            train_loss += loss.item() * batch_size

        train_loss /= samples_processed

        return samples_processed, train_loss

    def _nn_eval_epoch(self, loader):
        """Eval epoch."""
        self.nn_model.eval()
        val_loss = 0
        samples_processed = 0
        with torch.no_grad():
            for batch_samples in loader:

                # batch size x 1 x scatter dim x seqlen
                inputs = batch_samples['data'].permute(0, 2, 1).unsqueeze(1)
                # batch size x 1
                target_index = batch_samples['target_index']
                # batch size x feature_dim
                target = []
                for idx in target_index:
                    target += [torch.from_numpy(
                        self.wrmf.wrmf.item_factors[idx])]
                target = torch.stack(target)

                if self.USE_CUDA:
                    inputs = inputs.cuda()
                    target = target.cuda()

                # forward pass
                pred = self.nn_model(inputs)

                # compute val loss
                loss = self.nn_loss_func(pred, target)

                batch_size = inputs.size()[0]
                samples_processed += batch_size

                val_loss += loss.item() * batch_size

            val_loss /= samples_processed

            return samples_processed, val_loss

    def fit_nn(self, metadata_csv, save_dir):
        """
        Train the NN model.

        Args
            metadata_csv: path to the metadata_csv file.
            save_dir: directory to save nn_model

        """
        # Print settings to output file
        print("WRMF Name:{}\n\
               Output Size: {}\n\
               Dropout Rate: {}\n\
               Batch Size: {}\n\
               Learning Rate: {}\n\
               Beta One: {}\n\
               Beta Two: {}\n\
               EPS: {}\n\
               Weight Decay: {}\n\
               Num Epochs: {}\n\
               Save Dir: {}".format(
                   self.wrmf_name, self.output_size, self.dropout,
                   self.batch_size, self.lr, self.beta_one, self.beta_two,
                   self.nn_eps, self.weight_decay, self.num_epochs,
                   save_dir))

        self.models_dir = save_dir
        self.metadata_csv = metadata_csv

        # load data sets, create data loaders and initializer model
        self.load_nn_dataset(split='train')
        self.load_nn_dataset(split='val')

        train_loader = DataLoader(
            self.nn_train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)
        val_loader = DataLoader(
            self.nn_val_data, batch_size=self.batch_size, num_workers=4)

        self._init_nn()

        # Training loop
        train_loss = 0
        best_loss = float('inf')
        samples_processed = 0
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch

            # Training epoch
            if epoch > 0:
                sp, train_loss = self._nn_train_epoch(train_loader)
                samples_processed += sp

            # Validation epoch
            if epoch % 1 == 0:
                _, val_loss = self._nn_eval_epoch(val_loader)
                # report loss
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: \
                {}\tValidation Loss: {}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.nn_train_data)*self.num_epochs, train_loss,
                    val_loss))

            # Save model
            if val_loss < best_loss:
                best_loss = val_loss
                self.insert_nn_factors()
                self.save(models_dir=self.models_dir)

    def fit(self, triplets_txt, metadata_csv, save_dir):
        """
        Fit DCBR model.

        Args
            triplets_txt: path to the triplets_txt file.
            metadata_csv: path to the metadata_csv file.
            cf_save_dir: directory to save cf_model.
            nn_save_dir: directory to save nn_model.
        """
        self.fit_cf(triplets_txt)
        self.fit_nn(metadata_csv, save_dir)

    def insert_nn_factors(self):
        """Build learned item factor matrix."""
        self.load_nn_dataset(split='all')

        all_loader = DataLoader(
            self.nn_all_data, batch_size=self.batch_size, shuffle=False,
            num_workers=4)

        self.nn_model.eval()

        with torch.no_grad():
            for batch_samples in all_loader:
                # batch size x 1 x frequency dim x seqlen
                inputs = batch_samples['data'].permute(0, 2, 1).unsqueeze(1)
                # batch size x 1
                target_index = batch_samples['target_index']

                if self.USE_CUDA:
                    inputs = inputs.cuda()

                # forward pass
                pred = self.nn_model(inputs)

                # insert item_factors
                for i, idx in enumerate(target_index):
                    self.best_item_factors[idx] = pred[i]

    def insert_best_factors(self):
        """Insert best item factors."""
        self.wrmf.wrmf.item_factors = self.best_item_factors

    def score(self, users, split):
        """
        Score the model with AUC.

        Args
            users: list of user_ids
            split: split to score on
        """
        self.nn_model.eval()

        self.load_pred_dataset(split)

        auc = []
        for user_id in users:
            scores, targets = self.predict(user_id, split)

            if (scores is not None) and (targets is not None):
                auc += [roc_auc_score(targets, scores)]

        return np.mean(auc)

    def predict(self, user, split):
        """
        Predict for a user.

        Args
            user: a user id
            split: 'train' or 'test'
        """
        self.nn_pred_data.create_user_data(user)

        if self.nn_pred_data.user_has_songs:

            pred_loader = DataLoader(
                self.nn_pred_data, batch_size=self.batch_size, shuffle=False,
                num_workers=4)

            with torch.no_grad():
                scores = []
                targets = []
                for batch_samples in pred_loader:

                    u = []
                    for idx in batch_samples['user_index']:
                        u += [torch.from_numpy(
                            self.wrmf.wrmf.user_factors[idx])]
                    u = torch.stack(u)

                    i = []
                    for idx in batch_samples['item_index']:
                        i += [torch.from_numpy(
                            self.wrmf.wrmf.item_factors[idx])]
                    i = torch.stack(i)

                    target = batch_samples['target']

                    if self.USE_CUDA:
                        u = u.cuda()
                        i = i.cuda()
                        target = target.cuda()

                    score = F.cosine_similarity(u.float(), i.float())
                    scores += score.cpu().numpy().tolist()
                    targets += target.cpu().numpy().tolist()

            return scores, targets

        return None, None

    def save(self, models_dir):
        """
        Save models.

        Args
            models_dir: path to directory for saving model.
        """
        self.wrmf_name = "fact_"+str(self.wrmf.factors) + \
                         "_reg_"+str(self.wrmf.l2) + \
                         "_iter_"+str(self.wrmf.n_iter) + \
                         "_alpha_" + str(self.wrmf.alpha) + \
                         "_eps_" + str(self.wrmf.eps)

        if self.nn_model is not None:

            model_dir = "{}_CONV_drop_{}_lr_{}_b1_{}_b2_{}_wd_{}".format(
                self.wrmf_name, self.dropout, self.lr, self.beta_one,
                self.beta_two, self.weight_decay)

            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.nn_model.state_dict(),
                            'dcbr_dict': self.__dict__}, file)

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

        for (k, v) in checkpoint['dcbr_dict'].items():
            setattr(self, k, v)

        self._init_nn()
        self.nn_model.load_state_dict(checkpoint['state_dict'])


class DCUE(Trainer):

    """Class to train and evaluate DCUE model."""

    def __init__(self, feature_dim=100, batch_size=64, neg_batch_size=20,
                 u_embdim=300, margin=0.2, lr=0.00001, beta_one=0.9,
                 beta_two=0.99, eps=1e-8, weight_decay=0, num_epochs=100,
                 bn_momentum=0.5, data_type='mel', n_users=20000,
                 n_items=10000):
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
        self.data_type = data_type
        self.n_users = n_users
        self.n_items = n_items

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
                excluded_ids=excluded_ids)

        elif split == 'test':
            self.test_data = DCUEDataset(
                self.triplets_txt, self.metadata_csv, self.neg_batch_size,
                split, data_type=self.data_type, n_users=self.n_users,
                n_items=self.n_items, transform=transform,
                excluded_ids=excluded_ids)

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
            excluded_ids=excluded_ids)

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
            excluded_ids=excluded_ids)

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.dict_args = {'data_type': self.data_type,
                          'feature_dim': self.feature_dim,
                          'user_embdim': self.u_embdim,
                          'user_count': self.train_data.n_users,
                          'bn_momentum': self.bn_momentum}

        self.model = DCUENet(self.dict_args)

        self.loss_func = nn.MSELoss()
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
        k = 1

        for batch_samples in loader:

            # prepare training sample
            u = batch_samples['u']
            y = batch_samples['y']
            # batch size x 1 x seqdim x seqlen
            pos = batch_samples['pos'].unsqueeze(1).permute(0, 1, 3, 2)
            # batch size x neg batch size x 1 x seqdim x seqlen
            neg = batch_samples['neg'].unsqueeze(2).permute(0, 1, 2, 4, 3)

            if self.USE_CUDA:
                u = u.cuda()
                y = y.cuda()
                pos = pos.cuda()
                neg = neg.cuda()

            # forward pass
            self.model.zero_grad()
            preds = self.model(u, pos, neg)

            # backward pass
            loss = self.loss_func(preds, y.expand(-1, self.neg_batch_size))
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += pos.size()[0]
            losses_processed += preds.numel()

            train_loss += loss.item() * preds.numel()

            # report AUC mid epoch
            if samples_processed / (len(self.train_data) * 0.34) > k:
                # compute auc estimate.  Gives +/- approx 0.017 @ 95%
                # confidence w/ 20K users.
                val_auc = self._compute_auc('val', pct=0.025)

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\t\
                       Validation AUC: {}".format(
                           self.nn_epoch, self.num_epochs, samples_processed,
                           len(self.train_data), val_auc))
                k += 1

                self._update_best(val_auc)

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
                # batch size x 1 x seqdim x seqlen
                pos = batch_samples['pos'].unsqueeze(1).permute(0, 1, 3, 2)
                # batch size x neg batch size x 1 x seqdim x seqlen
                neg = batch_samples['neg'].unsqueeze(2).permute(0, 1, 2, 4, 3)

                if self.USE_CUDA:
                    u = u.cuda()
                    y = y.cuda()
                    pos = pos.cuda()
                    neg = neg.cuda()

                # clear gradients
                self.model.zero_grad()

                # forward pass
                preds = self.model(u, pos, neg)

                loss = self.loss_func(preds, y.expand(-1, self.neg_batch_size))

                # compute loss
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
               Data Type: {}\n\
               Num Users: {}\n\
               Num Items: {}\n\
               Save Dir: {}".format(
                   self.feature_dim, self.batch_size, self.neg_batch_size,
                   self.u_embdim, self.margin, self.lr, self.beta_one,
                   self.beta_two, self.eps, self.weight_decay, self.num_epochs,
                   self.data_type, self.n_users, self.n_items, save_dir))

        self.model_dir = save_dir
        self.metadata_csv = metadata_csv
        self.triplets_txt = triplets_txt

        print("Loading datasets...")
        self.load_dataset(split='train')
        self.load_dataset(split='val')
        self.load_item_dataset()

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=4)

        self._init_nn()

        # init training variables
        train_loss = 0
        samples_processed = 0
        self.best_auc = 0

        # train loop
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch
            if epoch > 0:
                print("Initializing train epoch...")
                sp, train_loss = self._train_epoch(train_loader)
                samples_processed += sp

            if epoch % 1 == 0:
                # compute loss
                print("Initializing val epoch...")
                _, val_loss = self._eval_epoch(val_loader)

                # compute auc estimate.  Gives +/- approx 0.017 @ 95%
                # confidence w/ 20K users.
                val_auc = self._compute_auc('val', pct=0.025)

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: \
                {}\tValidation Loss: {}\tValidation AUC: {}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    val_loss, val_auc))

            self._update_best(val_auc)

    def score(self, users, split):
        """
        Score the model with AUC.

        Args
            users: list of user_ids
            split: split to score on
        """
        self.model.eval()

        self.load_pred_dataset(split)

        auc = []
        for user_id in users:
            scores, targets = self.predict(user_id, split)

            if (scores is not None) and (targets is not None):
                auc += [roc_auc_score(targets, scores)]

        final_auc = np.mean(auc)
        return final_auc

    def predict(self, user, split):
        """
        Predict for a user.

        Args
            user: a user id
            split: 'train' or 'test'
        """
        self.pred_data.create_user_data(user)

        if self.pred_data.user_has_songs:

            pred_loader = DataLoader(
                self.pred_data, batch_size=self.batch_size, shuffle=False,
                num_workers=4)

            self.model.eval()

            with torch.no_grad():
                scores = []
                targets = []
                for batch_samples in pred_loader:

                    u = []
                    for idx in batch_samples['u']:
                        u += [self.user_factors[idx]]
                    u = torch.stack(u)

                    i = []
                    for idx in batch_samples['pos_song_idx']:
                        i += [self.item_factors[idx]]
                    i = torch.stack(i)

                    y = batch_samples['y']

                    if i.size()[0] > 1:
                        if self.USE_CUDA:
                            u = u.cuda()
                            i = i.cuda()

                        # forward pass
                        score = F.cosine_similarity(u, i)

                        scores += score.cpu().numpy().tolist()
                        targets += y.numpy().tolist()

            return scores, targets

        return None, None

    def insert_best_factors(self):
        """Insert the best factors for predictions."""
        self.item_factors = self.best_item_factors
        self.user_factors = self.best_user_factors

    def _update_best(self, val_auc):

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

    def _compute_auc(self, split, pct=0.025):
        print("Initializing AUC computation...")
        self._user_factors()
        self._item_factors()

        if split == 'train':
            users = list(self.train_data.dh.user_index.keys())
        elif split == 'val':
            users = list(self.val_data.dh.user_index.keys())
        elif split == 'test':
            users = list(self.test_data.dh.user_index.keys())

        n_users = len(users)
        users_sample = np.random.choice(users, int(n_users * 0.025))

        return self.score(users_sample, split)

    def _user_factors(self):
        """Create user factors matrix."""
        self.user_factors = torch.zeros([self.n_users, self.feature_dim])
        self.model.eval()
        with torch.no_grad():
            for i in list(self.item_data.dh.user_index.values()):
                emb_idx = torch.LongTensor([i])
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
                pos = batch_samples['pos'].unsqueeze(1).permute(0, 1, 3, 2)
                metadata_indexes = batch_samples['metadata_index']

                if self.USE_CUDA:
                    pos = pos.cuda()

                item_factor = self.model.conv(pos)

                for i, idx in enumerate(metadata_indexes):
                    self.item_factors[idx] = item_factor[i]

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = "DCUE_lr_{}_b1_{}_b2_{}_wd_{}_nu_{}_ni_{}".\
                format(self.lr, self.beta_one, self.beta_two,
                       self.weight_decay, self.n_users, self.n_items)

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
