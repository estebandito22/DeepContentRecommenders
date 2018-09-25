"""PyTorch classes for the language model component in DCUE."""

import torch
from torch import nn

from dc.dcue.embeddings.wordembedding import WordEmbeddings

from dc.dcue.languagemodels.recurrent import RecurrentNet


class LanguageModel(nn.Module):

    """Recurrent Net used on text inputs and audio features."""

    def __init__(self, dict_args):
        """
        Initialize LanguageModel.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(LanguageModel, self).__init__()
        self.feature_dim = dict_args["feature_dim"]
        self.conv_outsize = dict_args["conv_outsize"]
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]

        # rnn
        dict_args = {'conv_outsize': self.conv_outsize,
                     'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'hidden_size': self.hidden_size,
                     'dropout': self.dropout,
                     'batch_size': self.batch_size,
                     'attention': self.attention}
        self.rnn = RecurrentNet(dict_args)
        self.rnn.init_hidden(self.batch_size)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'word_embeddings': self.word_embeddings}
        self.word_embd = WordEmbeddings(dict_args)

        # item embedding
        self.fc = nn.Linear(
            self.hidden_size + self.conv_outsize, self.feature_dim)

    def _pos_forward(self, posseq, pos_convfeatvects, u_featvects):
        pos_convfeatvects = pos_convfeatvects.squeeze()

        # word embeddings
        seqembd = self.word_embd(posseq)

        # detach the hidden state of the rnn and perform forward pass on
        # rnn sequence.
        self.rnn.detach_hidden(pos_convfeatvects.size()[0])
        pos_rnn_log_probs, pos_rnn_hidden = self.rnn(
            seqembd, pos_convfeatvects)

        # concatenate final hidden state of rnn with conv features and
        # pass into FC layer to create final song embedding
        pos_featvects = torch.cat(
            [pos_rnn_hidden.squeeze(), pos_convfeatvects], dim=1)
        pos_featvects = self.fc(pos_featvects)

        return pos_featvects, pos_rnn_log_probs.permute(1, 2, 0)

    def _neg_forward(self, negseq, neg_convfeatvects, u_featvects):
        neg_convfeatvects = neg_convfeatvects.squeeze()

        # word embeddings
        seqlen, batch_size, neg_batch_size = negseq.size()
        negseq = negseq.view([seqlen, batch_size * neg_batch_size])
        seqembd = self.word_embd(negseq)

        # detach the hidden state of the rnn and perform forward pass on
        # rnn sequence.
        self.rnn.detach_hidden(batch_size * neg_batch_size)
        neg_rnn_log_probs, neg_rnn_hidden = self.rnn(
            seqembd, neg_convfeatvects)

        # concatenate final hidden state of rnn with conv features and
        # pass into FC layer to create final song embedding
        neg_featvects = torch.cat(
            [neg_rnn_hidden.squeeze(), neg_convfeatvects], dim=1)
        neg_featvects = self.fc(neg_featvects)

        # neg scores
        neg_featvects = neg_featvects.view(
            [batch_size, neg_batch_size, self.feature_dim])

        return neg_featvects, neg_rnn_log_probs.permute(1, 2, 0)

    def forward(self, u_featvects, posseq, pos_convfeatvects, negseq=None,
                neg_convfeatvects=None):
        """Forward pass."""
        if negseq is not None and neg_convfeatvects is not None:
            pos_featvects, pos_outputs = self._pos_forward(
                posseq, pos_convfeatvects, u_featvects)

            neg_featvects, neg_outputs = self._neg_forward(
                negseq, neg_convfeatvects, u_featvects)

        else:
            pos_featvects, pos_outputs = self._pos_forward(
                posseq, pos_convfeatvects, u_featvects)

            neg_featvects, neg_outputs = (None, None)

        return pos_featvects, pos_outputs, neg_featvects, neg_outputs
