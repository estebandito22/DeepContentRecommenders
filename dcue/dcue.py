"""
Deep Content User Embedding Neural Network.

Jongpil Lee, Kyungyun Lee, Jiyoung Park, Jangyeon Park, and Juhan Nam.
2016. Deep Content-User Embedding Model for Music Recommendation. In
Proceedings of DLRS 2018, Vancouver, Canada, October 6, 2018, 5 pages.
DOI: 10.1145/nnnnnnn.nnnnnnn
"""

import torch
import torch.nn as nn

from audiomodel import ConvNetScatter
from audiomodel import ConvNetMel

from userembedding import UserEmbeddings


class DCUE(nn.Module):

    """PyTorch class implementing DCUE Model."""

    def __init__(self, dict_args):
        """
        Initialize DCUE network.

        Takes a single argument dict_args that is a dictionary containing:

        input_type: 'scatter' or 'mel'
        feature_dim: The dimension of the embedded feature vectors for both
        users and audio.
        user_embdim: The dimension of the user lookup embedding.
        user_count: The count of users that will be embedded.
        """
        super(DCUE, self).__init__()

        # conv net attributes
        self.input_type = dict_args["input_type"]
        self.feature_dim = dict_args["feature_dim"]
        self.user_embdim = dict_args["user_embdim"]
        self.user_count = dict_args["user_count"]

        # convnet arguments
        dict_args = {'output_size': self.feature_dim}

        if self.input_type == 'scatter':
            self.conv = ConvNetScatter(dict_args)
        elif self.input_type == 'mel':
            self.conv = ConvNetMel(dict_args)

        # user embedding arguments
        dict_args = {'user_embdim': self.user_embdim,
                     'user_count': self.user_count,
                     'feature_dim': self.feature_dim}

        self.user_embd = UserEmbeddings(dict_args)

        self.sim = nn.CosineSimilarity(dim=1)

    def forward(self, u, pos, neg):
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

        # negative scores
        batch_size, neg_batch_size, channels, seqdim, seqlen = neg.size()
        neg = neg.view([batch_size * neg_batch_size, channels, seqdim, seqlen])
        neg_featvects = self.conv(neg)
        neg_featvects = neg_featvects.view(
            [batch_size, neg_batch_size, self.feature_dim])
        neg_scores = self.sim(
            u_featvects.unsqueeze(2), neg_featvects.permute(0, 2, 1))
        neg_scores = neg_scores.sum(dim=1)

        return pos_scores - neg_scores


if __name__ == '__main__':

    dict_argstest = {'output_size': 100}
    conv = ConvNetScatter(dict_argstest)
    sim = nn.CosineSimilarity(dim=1)

    utest = torch.ones([2, 100])*7

    postest = torch.ones([2, 1, 441, 17])
    postest[0] *= 3
    postest[1] *= 4

    pos_featvectstest = conv(postest)
    pos_scorestest = sim(utest, pos_featvectstest)

    negtest = torch.ones([2, 3, 1, 441, 17])
    negtest[0] *= 2
    negtest = negtest.view([6, 1, 441, 17])

    neg_featvectstest = conv(negtest)
    neg_featvectstest = neg_featvectstest.view([2, 3, 100])

    neg_scorestest = sim(
        utest.unsqueeze(2), neg_featvectstest.permute(0, 2, 1))

    neg_scorestest.sum(dim=1)
