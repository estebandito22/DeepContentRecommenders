"""Classes to train Deep Content Recommender Models."""

import os
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dc.nn.dcue import DCUE
from dc.optim.swats import Swats
from dc.dcue.dcuelm import DCUELMNet
from dc.datasets.dcuelmdataset import DCUELMDataset
from dc.datasets.dcuelmitemset import DCUELMItemset


class DCUELM(DCUE):

    """Trainer for DCUELM model."""

    def __init__(self, feature_dim=100, batch_size=64, neg_batch_size=20,
                 u_embdim=300, margin=0.2, optimize='adam', lr=0.01,
                 beta_one=0.9, beta_two=0.99, eps=1e-8, weight_decay=0,
                 num_epochs=100, bn_momentum=0.5, dropout=0, model_type='mel',
                 data_type='mel', n_users=20000, n_items=10000, eval_pct=0.025,
                 word_embdim=300, word_embeddings=None, hidden_size=512,
                 dropout_rnn=0, vocab_size=20000, attention=False,
                 loss_alpha=0.5, freeze_conv=True):
        """Initialize DCUELM trainer."""
        DCUE.__init__(self, feature_dim, batch_size, neg_batch_size, u_embdim,
                      margin, optimize, lr, beta_one, beta_two, eps,
                      weight_decay, num_epochs, bn_momentum, dropout,
                      model_type, data_type, n_users, n_items, eval_pct)

        self.word_embdim = word_embdim
        self.word_embeddings = word_embeddings
        self.hidden_size = hidden_size
        self.dropout_rnn = dropout_rnn
        self.vocab_size = vocab_size
        self.attention = attention
        self.loss_alpha = loss_alpha
        self.freeze_conv = freeze_conv

        self.side_loss_func = None

    def load_dataset(self, split, transform=None, excluded_ids=None):
        """
        Load datasets for NN.

        Args
            split: the split of the data to load
            transform: transformation to use for NN dataset
            excluded_ids: ids to exclude from the metadata_csv for NN model.
        """
        if split == 'train':
            self.train_data = DCUELMDataset(
                self.triplets_txt, self.metadata_csv, self.neg_batch_size,
                split, data_type=self.data_type, n_users=self.n_users,
                n_items=self.n_items, transform=transform,
                excluded_ids=excluded_ids, batch_songs=True)

        elif split == 'val':
            self.val_data = DCUELMDataset(
                self.triplets_txt, self.metadata_csv, self.neg_batch_size,
                split, data_type=self.data_type, n_users=self.n_users,
                n_items=self.n_items, transform=transform,
                excluded_ids=excluded_ids, random_seed=10, batch_songs=True)

        elif split == 'test':
            self.test_data = DCUELMDataset(
                self.triplets_txt, self.metadata_csv, self.neg_batch_size,
                split, data_type=self.data_type, n_users=self.n_users,
                n_items=self.n_items, transform=transform,
                excluded_ids=excluded_ids, random_seed=10, batch_songs=True)

    def load_item_dataset(self, transform=None, excluded_ids=None):
        """
        Load data for generating item_factors.

        Args:
            transform: transformation to use for NN dataset
            excluded_ids: ids to exclude from the metadata_csv for NN model.
        """
        self.item_data = DCUELMItemset(
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
                          'dropout': self.dropout,
                          'model_type': self.model_type,
                          'word_embdim': self.word_embdim,
                          'word_embeddings': self.word_embeddings,
                          'hidden_size': self.hidden_size,
                          'dropout_rnn': self.dropout_rnn,
                          'vocab_size': self.vocab_size,
                          'attention': self.attention,
                          'batch_size': self.batch_size}

        self.model = DCUELMNet(self.dict_args)

        if self.freeze_conv:
            for param in self.model.conv.parameters():
                param.requires_grad = False

        self.loss_func = nn.HingeEmbeddingLoss(margin=self.margin)
        self.side_loss_func = nn.NLLLoss()
        if self.optimize == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), self.lr,
                (self.beta_one, self.beta_two),
                self.eps, self.weight_decay)
        elif self.optimize == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), self.lr, self.beta_one,
                weight_decay=self.weight_decay, nesterov=True)
        elif self.optimize == 'swats':
            self.optimizer = Swats(
                self.model.parameters(), self.lr,
                (self.beta_one, self.beta_two),
                self.eps, self.weight_decay)

        self.scheduler = MultiStepLR(
            self.optimizer, milestones=[4, 9], gamma=0.1)

        if self.USE_CUDA:
            self.model = self.model.cuda()
            self.loss_func = self.loss_func.cuda()
            self.side_loss_func = self.side_loss_func.cuda()

    def _train_epoch(self, loaders):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        train_side_loss = 0
        samples_processed = 0
        losses_processed = 0

        for i, loader in enumerate(loaders):
            loader.dataset.dataset.load_song_batch(i)
            for batch_samples in tqdm(loader):

                # prepare training sample
                u = batch_samples['u']
                y = batch_samples['y']
                # batch size x num words
                pos_t = batch_samples['t'].t().contiguous()
                # batch size x seqdim x seqlen
                pos = batch_samples['X']
                # batch size x neg batch size x seqdim x seqlen
                neg, neg_t = self._withinbatch_negsample(pos, pos_t)

                if self.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    pos = pos.unsqueeze(1)
                    # batch size x neg batch size x 1 x seqdim x seqlen
                    neg = neg.unsqueeze(2)

                if self.USE_CUDA:
                    u = u.cuda()
                    y = y.cuda()
                    pos = pos.cuda()
                    pos_t = pos_t.cuda()
                    neg = neg.cuda()
                    neg_t = neg_t.cuda()

                # forward pass
                self.model.zero_grad()
                scores, log_probs = self.model(u, pos, pos_t, neg, neg_t)

                # backward pass hinge loss
                loss_1 = self.loss_func(scores, y)
                # loss.backward(retain_graph=True)

                # backward pass NLL loss
                nbs = u.size()[0] * self.neg_batch_size
                y_t = torch.cat([pos_t.t(), neg_t.view(-1, nbs).t()])
                loss_2 = self.side_loss_func(log_probs, y_t)

                loss = self.loss_alpha * loss_1 + (1 - self.loss_alpha) * loss_2
                loss.backward()

                # optimization step
                self.optimizer.step()

                # compute train loss
                samples_processed += pos.size()[0]
                losses_processed += scores.numel()

                train_loss += loss_1.item() * scores.numel()
                train_side_loss += loss_2.item() * scores.numel()

            train_loss /= losses_processed
            train_side_loss /= losses_processed

        return samples_processed, train_loss, train_side_loss

    def _eval_epoch(self, loaders):
        """Eval epoch."""
        self.model.eval()
        val_loss = 0
        val_side_loss = 0
        samples_processed = 0
        losses_processed = 0

        with torch.no_grad():
            for i, loader in enumerate(loaders):
                loader.dataset.dataset.load_song_batch(i)
                for batch_samples in tqdm(loader):

                    # prepare training sample
                    u = batch_samples['u']
                    y = batch_samples['y']
                    # batch size x num words
                    pos_t = batch_samples['t'].t().contiguous()
                    # batch size x seqdim x seqlen
                    pos = batch_samples['X']
                    # batch size x neg batch size x seqdim x seqlen
                    neg, neg_t = self._withinbatch_negsample(pos, pos_t)

                    if self.model_type.find('2d') > -1:
                        # batch size x 1 x seqdim x seqlen
                        pos = pos.unsqueeze(1)
                        # batch size x neg batch size x 1 x seqdim x seqlen
                        neg = neg.unsqueeze(2)

                    if self.USE_CUDA:
                        u = u.cuda()
                        y = y.cuda()
                        pos = pos.cuda()
                        pos_t = pos_t.cuda()
                        neg = neg.cuda()
                        neg_t = neg_t.cuda()

                    # forward pass
                    scores, log_probs = self.model(u, pos, pos_t, neg, neg_t)

                    # compute loss
                    loss_1 = self.loss_func(scores, y)

                    # compute side loss
                    nbs = u.size()[0] * self.neg_batch_size
                    y_t = torch.cat([pos_t.t(), neg_t.view(-1, nbs).t()])
                    loss_2 = self.side_loss_func(log_probs, y_t)

                    samples_processed += pos.size()[0]
                    losses_processed += scores.numel()

                    val_loss += loss_1.item() * scores.numel()
                    val_side_loss += loss_2.item() * scores.numel()

                val_loss /= losses_processed
                val_side_loss /= losses_processed

        return samples_processed, val_loss, val_side_loss

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
               Dropout: {}\n\
               Optimizer: {}\n\
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
               Hidden Size: {}\n\
               Dropout RNN: {}\n\
               Attention: {}\n\
               Save Dir: {}".format(
                   self.feature_dim, self.batch_size, self.neg_batch_size,
                   self.u_embdim, self.margin, self.dropout, self.optimize,
                   self.lr, self.beta_one, self.beta_two, self.eps,
                   self.weight_decay, self.num_epochs, self.model_type,
                   self.data_type, self.n_users, self.n_items,
                   self.hidden_size, self.dropout_rnn, self.attention,
                   save_dir))

        self.model_dir = save_dir
        self.metadata_csv = metadata_csv
        self.triplets_txt = triplets_txt

        print("Loading datasets...")
        self.load_dataset(split='train')
        self.load_dataset(split='val')
        self.load_item_dataset()
        self.load_pred_dataset(split='val')

        pred_loader = DataLoader(
            self.pred_data, batch_size=1024, shuffle=True, num_workers=4)

        self._init_nn()

        # init training variables
        train_loss = 0
        train_sloss = 0
        samples_processed = 0
        self.best_auc = 0
        self.best_val_loss = float('inf')

        # train loop
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch
            if epoch > 0:
                print("Initializing train epoch...")
                start = datetime.datetime.now()
                train_loaders = self._batch_loaders(self.train_data)
                sp, train_loss, train_sloss = self._train_epoch(train_loaders)
                samples_processed += sp
                end = datetime.datetime.now()
                print("Train epoch processed in {}".format(end-start))

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss:{}\t\
                Train Side Loss:{}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    train_sloss))

            if epoch % 5 == 0:
                # compute loss
                print("Initializing val epoch...")
                start = datetime.datetime.now()
                val_loaders = self._batch_loaders(self.val_data)
                _, val_loss, val_sloss = self._eval_epoch(val_loaders)
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
                {}\tValidation Loss: {}\tValidation AUC: {}\t\
                Val Side Loss: {}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    val_loss, val_auc, val_sloss))

            self._update_best(val_auc, val_loss)

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
                t = batch_samples['t'].t().contiguous()
                if self.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    X = X.unsqueeze(1)
                metadata_indexes = batch_samples['metadata_index']

                if self.USE_CUDA:
                    X = X.cuda()
                    t = t.cuda()

                conv_featvect = self.model.conv(X)
                item_factor, _, _, _ = self.model.lm(t, conv_featvect)

                for i, idx in enumerate(metadata_indexes):
                    self.item_factors[idx] = item_factor[i]

    def _withinbatch_negsample(self, song_batch, seq_batch):
        batch_size, seqdim, seqlen = song_batch.size()
        neg = torch.zeros([batch_size, self.neg_batch_size, seqdim, seqlen])

        seqlen, _ = seq_batch.size()
        neg_t = torch.zeros([seqlen, batch_size, self.neg_batch_size]).long()

        for i in range(batch_size):
            indexes = [x for x in range(0, i)] + \
                      [x for x in range(i+1, batch_size)]
            for j in range(self.neg_batch_size):
                rand_idx = np.random.choice(indexes)
                neg[i][j].copy_(song_batch[rand_idx])
                neg_t[:, i, j].copy_(seq_batch[:, rand_idx])

        return neg, neg_t

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = "DCUELM_do_{}_op_{}_lr_{}_b1_{}_b2_{}_wd_{}_nu_{}\
            _ni_{}_mt_{}_hs_{}_dr_{}_at_{}".\
                format(self.dropout, self.optimize, self.lr, self.beta_one,
                       self.beta_two, self.weight_decay, self.n_users,
                       self.n_items, self.model_type, self.hidden_size,
                       self.dropout_rnn, self.attention)

            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)
