"""Deep Content User Embedding - Language Model Neural Network."""

import torch
import torch.nn as nn

from dc.dcue.audiomodels.convmel1d import ConvNetMel1D
from dc.dcue.audiomodels.convmel2d import ConvNetMel2D
from dc.dcue.audiomodels.l3mel2d import L3NetMel2D
from dc.dcue.audiomodels.l3mel2dsmall import L3NetMel2DSmall

from dc.dcue.audiomodels.convmeltrunc2d import ConvNetMelTrunc2D
from dc.dcue.audiomodels.l3meltrunc2d import L3NetMelTrunc2D

from dc.dcue.languagemodels.lm import LanguageModel

from dc.dcue.embeddings.userembedding import UserEmbeddings


class DCUELMNet(nn.Module):

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
        super(DCUELMNet, self).__init__()

        # attributes
        self.data_type = dict_args["data_type"]
        self.feature_dim = dict_args["feature_dim"]
        self.user_embdim = dict_args["user_embdim"]
        self.user_count = dict_args["user_count"]
        self.bn_momentum = dict_args["bn_momentum"]
        self.dropout = dict_args["dropout"]
        self.model_type = dict_args["model_type"]
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout_rnn = dict_args["dropout_rnn"]
        self.vocab_size = dict_args["vocab_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]

        # convnet arguments
        dict_args = {'output_size': self.feature_dim,
                     'bn_momentum': self.bn_momentum,
                     'dropout': self.dropout}

        if self.model_type == 'mel1d':
            self.conv = ConvNetMel1D(dict_args)
            self.conv_outsize = self.feature_dim
        elif self.model_type == 'mel2d':
            self.conv = ConvNetMel2D(dict_args)
            self.conv_outsize = self.feature_dim
        elif self.model_type == 'l3mel2d':
            self.conv = L3NetMel2D(dict_args)
            self.conv_outsize = self.feature_dim
        elif self.model_type == 'l3mel2dsmall':
            self.conv = L3NetMel2DSmall(dict_args)
            self.conv_outsize = self.feature_dim
        elif self.model_type == 'l3meltrunc2d':
            self.conv = L3NetMelTrunc2D(dict_args)
            self.conv_outsize = self.conv.layer_outsize4[0]
        elif self.model_type == 'meltrunc2d':
            self.conv = ConvNetMelTrunc2D(dict_args)
            self.conv_outsize = self.conv.layer_outsize5[0]

        # user embedding arguments
        dict_args = {'user_embdim': self.user_embdim,
                     'user_count': self.user_count,
                     'feature_dim': self.feature_dim}

        self.user_embd = UserEmbeddings(dict_args)

        # language model arguments
        dict_args = {'feature_dim': self.feature_dim,
                     'conv_outsize': self.conv_outsize,
                     'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'hidden_size': self.hidden_size,
                     'dropout': self.dropout_rnn,
                     'vocab_size': self.vocab_size,
                     'batch_size': self.batch_size,
                     'attention': self.attention}

        self.lm = LanguageModel(dict_args)

        self.sim = nn.CosineSimilarity(dim=1)

    def forward(self, u, pos, posseq, neg=None, negseq=None):
        """
        Forward pass.

        Forward computes positive scores using u feature vector and ConvNet
        on positive sample and negative scores using u feature vector and
        ConvNet on randomly sampled negative examples.
        """
        # user features
        u_featvects = self.user_embd(u)

        # positive conv features
        pos_convfeatvects = self.conv(pos)

        # negative conv features
        if neg is not None and negseq is not None:
            if self.model_type.find('1d') > -1:
                batch_size, neg_batch_size, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, seqdim, seqlen])
            elif self.model_type.find('2d') > -1:
                batch_size, neg_batch_size, chan, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, chan, seqdim, seqlen])
            neg_convfeatvects = self.conv(neg)
        else:
            neg_convfeatvects = None

        # language model
        pos_featvects, pos_outputs, neg_featvects, neg_outputs = self.lm(
            posseq, pos_convfeatvects, negseq, neg_convfeatvects)

        # pos and neg scores
        pos_scores = self.sim(u_featvects, pos_featvects)
        if neg_featvects is not None:
            neg_scores = self.sim(
                u_featvects.unsqueeze(2), neg_featvects.permute(0, 2, 1))
        else:
            neg_scores = 0

        # language model outputs to be passed to NLLLoss
        if neg_outputs is not None:
            outputs = outputs = torch.cat([pos_outputs, neg_outputs])
        else:
            outputs = pos_outputs

        # difference of scores to be passed to Hinge Loss
        scores = pos_scores.view(pos_scores.size()[0], 1) - neg_scores

        return scores, outputs


if __name__ == '__main__':

    dict_argstest = {'data_type': 'mel',
                     'feature_dim': 100,
                     'user_embdim': 300,
                     'user_count': 100,
                     'bn_momentum': 0.5,
                     'dropout': 0,
                     'model_type': 'l3meltrunc2d',
                     'word_embdim': 300,
                     'word_embeddings': None,
                     'hidden_size': 256,
                     'dropout_rnn': 0,
                     'vocab_size': 20,
                     'batch_size': 2,
                     'attention': False}

    dcuelm = DCUELMNet(dict_argstest)

    yhinge = torch.ones([2, 1])*-1

    utest = torch.randint(0, 99, [2]).long()

    postest = torch.ones([2, 1, 128, 44])
    postest[0] *= 3
    postest[1] *= 4

    negtest = torch.ones([2, 3, 1, 128, 44])
    negtest[0] *= 3

    posseqtest = torch.randint(0, 19, [10, 2]).long()
    negseqtest = torch.randint(0, 19, [10, 2, 3]).long()

    hloss = nn.HingeEmbeddingLoss(0.2)
    nloss = nn.NLLLoss()

    scorestest, outputstest = dcuelm(utest, postest, posseqtest, negtest, negseqtest)

    hloss(scorestest, yhinge.expand(-1, 3))
    nloss(outputstest, torch.cat([posseqtest.t(), negseqtest.view(-1, 6).t()]))
