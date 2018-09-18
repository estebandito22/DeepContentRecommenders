"""
Deep Content User Embedding Neural Network.

Jongpil Lee, Kyungyun Lee, Jiyoung Park, Jangyeon Park, and Juhan Nam.
2016. Deep Content-User Embedding Model for Music Recommendation. In
Proceedings of DLRS 2018, Vancouver, Canada, October 6, 2018, 5 pages.
DOI: 10.1145/nnnnnnn.nnnnnnn
"""

import torch
import torch.nn as nn

from dc.dcue.audiomodel import ConvNetScatter
from dc.dcue.audiomodel import ConvNetMel1D
from dc.dcue.audiomodel import ConvNetMel2D

from dc.dcue.userembedding import UserEmbeddings


class DCUENet(nn.Module):

    """PyTorch class implementing DCUE Model."""

    def __init__(self, dict_args):
        """
        Initialize DCUE network.

        Takes a single argument dict_args that is a dictionary containing:

        data_type: 'scatter' or 'mel'
        feature_dim: The dimension of the embedded feature vectors for both
        users and audio.
        user_embdim: The dimension of the user lookup embedding.
        user_count: The count of users that will be embedded.
        """
        super(DCUENet, self).__init__()

        # conv net attributes
        self.data_type = dict_args["data_type"]
        self.feature_dim = dict_args["feature_dim"]
        self.user_embdim = dict_args["user_embdim"]
        self.user_count = dict_args["user_count"]
        self.bn_momentum = dict_args["bn_momentum"]
        self.dropout = dict_args["dropout"]
        self.model_type = dict_args["model_type"]

        # convnet arguments
        dict_args = {'output_size': self.feature_dim,
                     'bn_momentum': self.bn_momentum,
                     'dropout': self.dropout}

        if self.model_type == 'scatter':
            self.conv = ConvNetScatter(dict_args)
        elif self.model_type == 'mel1d':
            self.conv = ConvNetMel1D(dict_args)
        elif self.model_type == 'mel2d':
            self.conv = ConvNetMel2D(dict_args)

        # user embedding arguments
        dict_args = {'user_embdim': self.user_embdim,
                     'user_count': self.user_count,
                     'feature_dim': self.feature_dim}

        self.user_embd = UserEmbeddings(dict_args)

        self.sim = nn.CosineSimilarity(dim=1)

    def forward(self, u, pos, neg=None):
        """
        Forward pass.

        Forward computes positive scores using u feature vector and ConvNet
        on positive sample and negative scores using u feature vector and
        ConvNet on randomly sampled negative examples.
        """
        u_featvects = self.user_embd(u)

        # positive score
        pos_featvects = self.conv(pos)
        pos_scores = self.sim(u_featvects, pos_featvects)

        if neg is not None:
            # negative scores
            if self.model_type.find('1d') > -1:
                batch_size, neg_batch_size, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, seqdim, seqlen])
            elif self.model_type.find('2d') > -1:
                batch_size, neg_batch_size, chan, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, chan, seqdim, seqlen])
            neg_featvects = self.conv(neg)
            neg_featvects = neg_featvects.view(
                [batch_size, neg_batch_size, self.feature_dim])
            neg_scores = self.sim(
                u_featvects.unsqueeze(2), neg_featvects.permute(0, 2, 1))
        else:
            neg_scores = 0

        scores = pos_scores.view(pos_scores.size()[0], 1) - neg_scores

        return scores


if __name__ == '__main__':

    truth = torch.ones([2, 1])*-1

    dict_argstest = {'output_size': 100, 'bn_momentum': 0.5}
    conv = ConvNetMel(dict_argstest)
    sim = nn.CosineSimilarity(dim=1)

    utest = torch.ones([2, 100])*7
    utest[0] *= 2

    postest = torch.ones([2, 128, 44])
    postest[0] *= 3
    postest[1] *= 4

    pos_featvectstest = conv(postest)
    pos_scorestest = sim(utest, pos_featvectstest)

    negtest = torch.ones([2, 3, 128, 44])
    negtest[0] *= 3
    negtest = negtest.view([6, 128, 44])

    neg_featvectstest = conv(negtest)
    neg_featvectstest = neg_featvectstest.view([2, 3, 100])

    neg_scorestest = sim(
        utest.unsqueeze(2), neg_featvectstest.permute(0, 2, 1))

    scorestest = pos_scorestest.view(
        pos_scorestest.size()[0], 1) - neg_scorestest

    scorestest[1, 0] = 0
    scorestest[1, 1] = 2
    scorestest[1, 2] = 2
    scorestest[0, 0] = -1
    scorestest[0, 1] = 2
    scorestest[0, 2] = 2

    loss = nn.HingeEmbeddingLoss(0.2)

    loss(scorestest, truth.expand(-1, 3))
