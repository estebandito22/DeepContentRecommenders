"""Classes to train Deep Content Recommender Models."""

from abc import ABC, abstractmethod

import os

import numpy as np

import torch
import torch.nn as nn
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

from dc.pytorchdatasets import ToTensor
from dc.pytorchdatasets import RandomSample

from dc.dcbr.nn.audiomodel import ConvNetMel
from dc.dcbr.nn.audiomodel import ConvNetScatter

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
        self.cf_model_dir = None
        self.dh = None
        self.wrmf = None

        # additional nn attributes
        self.metadata_csv = None
        self.nn_model_dir = None
        self.factors_csv = None
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

        # combined model attributes
        self.user_factors = None
        self.item_factors = None

        if torch.cuda.is_available():
            self.USE_CUDA = True

    def load_cf_dataset(self):
        """Load datasets for CF models."""
        # cf data
        if self.dh is None:
            self.dh = CFDataHandler(self.triplets_txt)
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
                    self.metadata_csv, self.factors_csv, split,
                    mode='dev', return_id=False, transform=transform,
                    excluded_ids=excluded_ids, data_type=self.data_type)

        elif split == 'val':
            if self.wrmf is not None:
                # nn data
                self.nn_val_data = DCBRDataset(
                    self.metadata_csv, self.factors_csv, split,
                    mode='dev', return_id=False, transform=transform,
                    excluded_ids=excluded_ids, data_type=self.data_type)

        elif split == 'test':
            if self.wrmf is not None:
                # nn data
                self.nn_test_data = DCBRDataset(
                    self.metadata_csv, self.factors_csv, split,
                    mode='dev', return_id=False, transform=transform,
                    excluded_ids=excluded_ids, data_type=self.data_type)

        elif split == 'all':
            if self.wrmf is not None:
                # nn data
                self.nn_all_data = DCBRDataset(
                    self.metadata_csv, self.factors_csv, split,
                    mode='inference', return_id=False, transform=transform,
                    excluded_ids=excluded_ids, data_type=self.data_type)

    def load_pred_dataset(self, split, transform=None, excluded_ids=None):
        """Load data for predicting."""
        if self.wrmf is not None:
            self.nn_pred_data = DCBRPredset(
                self.metadata_csv, self.u_factors_csv, self.dh,
                split=split, transform=transform,
                excluded_ids=excluded_ids, data_type=self.data_type)

    def fit_cf(self, triplets_txt, save_dir):
        """
        Train CF model.

        Args
            triplets_txt: path to the triplets_txt file.
            save_dir: directory to save model.
        """
        self.triplets_txt = triplets_txt
        self.cf_model_dir = save_dir

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

        self.save(cf_model_dir=self.cf_model_dir)
        self.user_factors = torch.from_numpy(self.wrmf.wrmf.user_factors)

    def _init_nn(self):
        """Initialize the nn model for training."""
        # Initialize model and optimizer
        self.nn_dict_args = {
            'output_size': self.output_size,
            'dropout': self.dropout,
            'init_fc_bias': self.nn_train_data.target_data.mean(0),
            'bn_momentum': self.bn_momentum}

        if self.data_type == 'mel':
            self.nn_model = ConvNetMel(self.nn_dict_args)
        elif self.data_type == 'scatter':
            self.nn_model = ConvNetScatter(self.nn_dict_args)

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
            # batch size x factors dim
            target = batch_samples['target']

            if self.USE_CUDA:
                inputs = inputs.cuda()
                target = target.cuda()

            # clear gradients
            self.nn_model.zero_grad()

            # forward pass
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
                # batch size x factors dim
                target = batch_samples['target']

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
        print("Factors File:{}\n\
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
                   self.factors_csv, self.output_size, self.dropout,
                   self.batch_size, self.lr, self.beta_one, self.beta_two,
                   self.nn_eps, self.weight_decay, self.num_epochs,
                   save_dir))

        self.nn_model_dir = save_dir
        self.metadata_csv = metadata_csv

        # load data sets, create data loaders and initializer model
        if self.data_type == 'mel':
            composed = Compose([ToTensor(), RandomSample(44, 0)])
        elif self.data_type == 'scatter':
            composed = Compose([ToTensor(), RandomSample(17, 0)])

        self.load_nn_dataset(split='train', transform=composed)
        self.load_nn_dataset(split='val', transform=composed)

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
                self.save(nn_model_dir=self.nn_model_dir)

    def fit(self, triplets_txt, metadata_csv, cf_save_dir, nn_save_dir):
        """
        Fit DCBR model.

        Args
            triplets_txt: path to the triplets_txt file.
            metadata_csv: path to the metadata_csv file.
            cf_save_dir: directory to save cf_model.
            nn_save_dir: directory to save nn_model.
        """
        self.fit_cf(triplets_txt, cf_save_dir)
        self.fit_nn(metadata_csv, nn_save_dir)

    def _make_item_factors_nn(self):
        """Build learned item factor matrix."""
        if self.data_type == 'mel':
            composed = Compose([ToTensor(), RandomSample(44, 0)])
        elif self.data_type == 'scatter':
            composed = Compose([ToTensor(), RandomSample(17, 0)])

        self.load_nn_dataset(split='all', transform=composed)

        all_loader = DataLoader(
            self.nn_all_data, batch_size=self.batch_size, shuffle=False,
            num_workers=4)

        self.nn_model.eval()

        self.item_factors = torch.zeros(
            [len(self.nn_all_data), self.factors])

        samples_processed = 0
        for batch_samples in all_loader:
            # batch size x 1 x frequency dim x seqlen
            inputs = batch_samples['data'].permute(0, 2, 1).unsqueeze(1)

            if self.USE_CUDA:
                inputs = inputs.cuda()

            # forward pass
            pred = self.nn_model(inputs)

            # Insert item factors
            for i in range(pred.size()[0]):
                self.item_factors[i + samples_processed] = pred[i]

            batch_size = inputs.size()[0]
            samples_processed += batch_size

    def score(self, users, split):
        """
        Score the model with AUC.

        Args
            users: list of user_ids
            split: split to score on
        """
        self.nn_model.eval()

        if self.data_type == 'mel':
            transformer = RandomSample(44, 0)
        elif self.data_type == 'scatter':
            transformer = RandomSample(17, 0)

        self.load_pred_dataset(split, transform=transformer)
        self._make_item_factors_nn()

        auc = []
        for user_id in users:
            scores, targets = self.predict(user_id, split)

            if (scores is not None) and (targets is not None):
                user_auc = roc_auc_score(targets, scores)
                auc += [roc_auc_score(targets, scores)]
                print("User AUC: {}".format(user_auc))

        final_auc = np.mean(auc)
        print("Overall AUC {}".format(final_auc))

    def predict(self, user, split):
        """
        Predict for a user.

        Args
            user: a user id
            split: 'train' or 'test'
        """
        i = self.dh.user_index[user]
        self.nn_pred_data.create_user_data(i)

        if self.nn_pred_data.user_has_songs:

            pred_loader = DataLoader(
                self.nn_pred_data, batch_size=self.batch_size, shuffle=False,
                num_workers=4)

            with torch.no_grad():
                scores = []
                targets = []
                for batch_samples in pred_loader:

                    u = batch_samples['u']
                    indexes = batch_samples['i']
                    items = self.item_factors[indexes]
                    target = batch_samples['target']

                    if self.USE_CUDA:
                        u = u.cuda()
                        items = items.cuda()
                        target = target.cuda()

                    score = torch.mm(u.t().float(), items.float())
                    scores += score.cpu().numpy().tolist()
                    targets += target.cpu().numpy().tolist()

            return scores, targets

        return None, None

    # def predict(self, user, split):
    #     """
    #     Predict for a user.
    #
    #     Args
    #         user: a user id
    #         split: 'train' or 'test'
    #     """
    #     i = self.dh.user_index[user]
    #     self.nn_pred_data.create_user_data(i)
    #
    #     if self.nn_pred_data.user_has_songs:
    #
    #         pred_loader = DataLoader(
    #             self.nn_pred_data, batch_size=self.batch_size, shuffle=False,
    #             num_workers=4)
    #
    #         # self.nn_model.eval()
    #
    #         with torch.no_grad():
    #             scores = []
    #             targets = []
    #             for i, batch_samples in enumerate(pred_loader):
    #                 if i == np.ceil(len(self.nn_pred_data) * 0.10):
    #                     self.nn_model.eval()
    #
    #                 u = batch_samples['u']
    #                 inputs = batch_samples[
    #                     'data'].permute(0, 2, 1).unsqueeze(1)
    #                 target = batch_samples['target']
    #
    #                 if self.USE_CUDA:
    #                     u = u.cuda()
    #                     inputs = inputs.cuda()
    #                     target = target.cuda()
    #
    #                 preds = self.nn_model(inputs)
    #                 score = torch.mm(u.t().float(), preds.float())
    #
    #                 scores += score.cpu().numpy().tolist()
    #                 targets += target.cpu().numpy().tolist()
    #
    #             # print("Scores: {}\n\
    #             #        Targets: {}".format(scores, targets))
    #             # scores = torch.stack(scores)
    #             # targets = torch.stack(targets)
    #
    #         return scores, targets
    #
    #     return None, None

    def save(self, cf_model_dir=None, nn_model_dir=None):
        """
        Save models.

        Args
            cf_model_dir: path to directory for saving CF model factors.
            nn_model_dir: path to directory for saving NN models.
        """
        if (self.wrmf is not None) and (cf_model_dir is not None):
            # save learned item factors and summary
            fname = "item_"+str(self.wrmf.factors) + \
                    "_reg_"+str(self.wrmf.l2) + \
                    "_iter_"+str(self.wrmf.n_iter) + \
                    "_alpha_" + str(self.wrmf.alpha) + \
                    "_eps_" + str(self.wrmf.eps) + ".csv"

            f = os.path.join(cf_model_dir, fname)
            np.savetxt(f, self.wrmf.wrmf.item_factors)
            self.factors_csv = f

            fname = "user_"+str(self.wrmf.factors) + \
                    "_reg_"+str(self.wrmf.l2) + \
                    "_iter_"+str(self.wrmf.n_iter) + \
                    "_alpha_" + str(self.wrmf.alpha) + \
                    "_eps_" + str(self.wrmf.eps) + ".csv"

            f = os.path.join(cf_model_dir, fname)
            np.savetxt(f, self.wrmf.wrmf.user_factors)
            self.u_factors_csv = f

        if (self.nn_model is not None) and (nn_model_dir is not None):

            model_dir = "{}_CONV_drop_{}_lr_{}_b1_{}_b2_{}_wd_{}".format(
                self.factors_csv.rsplit(".", 1)[0], self.dropout,
                self.lr, self.beta_one, self.beta_two, self.weight_decay)

            if not os.path.isdir(os.path.join(nn_model_dir, model_dir)):
                os.makedirs(os.path.join(nn_model_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(nn_model_dir, model_dir, filename)
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

        # Model attributes
        self.model = None
        self.optimizer = None
        self.loss_func = None
        self.dict_args = None
        self.nn_epoch = None

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
            data_type=self.data_type, transform=transform,
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

            # backward pass
            loss = self.loss_func(preds, y)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            batch_size = pos.size()[0]
            preds_size = preds.numel()

            samples_processed += batch_size
            losses_processed += preds_size

            train_loss += loss.item() * preds_size

        train_loss /= losses_processed

        return samples_processed, train_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        # self.model.eval()
        val_loss = 0
        samples_processed = 0
        losses_processed = 0
        self.model.eval()
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

                loss = self.loss_func(preds, y)

                # compute loss
                batch_size = pos.size()[0]
                preds_size = preds.numel()

                samples_processed += batch_size
                losses_processed += preds_size

                val_loss += loss.item() * preds_size

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
               Input Type: {}\n\
               Save Dir: {}".format(
                   self.feature_dim, self.batch_size, self.neg_batch_size,
                   self.u_embdim, self.margin, self.lr, self.beta_one,
                   self.beta_two, self.eps, self.weight_decay, self.num_epochs,
                   self.input_type, save_dir))

        self.model_dir = save_dir
        self.metadata_csv = metadata_csv
        self.triplets_txt = triplets_txt

        if self.data_type == 'mel':
            composed = Compose(
                [ToTensor(), RandomSample(44, 0), RandomSample(44, 1)])
        elif self.data_type == 'scatter':
            composed = Compose(
                [ToTensor(), RandomSample(17, 0), RandomSample(17, 1)])

        self.load_nn_dataset(split='train', transform=composed)
        self.load_nn_dataset(split='val', transform=composed)

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=4)

        self._init_nn()

        train_loss = 0
        best_loss = float('inf')
        samples_processed = 0
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch
            if epoch > 0:
                sp, train_loss = self._nn_train_epoch(train_loader)
                samples_processed += sp

            if epoch % 1 == 0:
                _, val_loss = self._nn_eval_epoch(val_loader)
                # report loss
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: \
                {}\tValidation Loss: {}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    val_loss))

            if val_loss < best_loss:
                best_loss = val_loss
                self.save(models_dir=self.model_dir)

    def score(self, users, split):
        """
        Score the model with AUC.

        Args
            users: list of user_ids
            split: split to score on
        """
        self.model.eval()

        if self.data_type == 'mel':
            composed = Compose([RandomSample(44, 0), RandomSample(44, 1)])
        elif self.data_type == 'scatter':
            composed = Compose([RandomSample(17, 0), RandomSample(17, 1)])

        self.load_pred_dataset(split, transform=composed)

        auc = []
        for user_id in users:
            scores, targets = self.predict(user_id, split)

            if (scores is not None) and (targets is not None):
                user_auc = roc_auc_score(targets, scores)
                auc += [roc_auc_score(targets, scores)]
                print("User AUC: {}".format(user_auc))

        final_auc = np.mean(auc)
        print("Overall AUC {}".format(final_auc))

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

                    u = batch_samples['u']
                    y = batch_samples['y']
                    # batch size x 1 x seqdim x seqlen
                    pos = batch_samples['pos'].unsqueeze(1).permute(0, 1, 3, 2)

                    if self.USE_CUDA:
                        u = u.cuda()
                        y = y.cuda()
                        pos = pos.cuda()

                    # forward pass
                    preds = self.model(u, pos)

                    scores += preds.cpu().numpy().tolist()
                    targets += y.cpu().numpy().tolist()

            return scores, targets

        return None, None

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.nn_model is not None) and (models_dir is not None):

            model_dir = "{}_CONV_drop_{}_lr_{}_b1_{}_b2_{}_wd_{}".format(
                self.factors_csv.rsplit(".", 1)[0], self.dropout,
                self.lr, self.beta_one, self.beta_two, self.weight_decay)

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
