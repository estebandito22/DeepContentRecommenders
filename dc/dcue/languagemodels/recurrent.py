"""PyTorch classes for the language model component in DCUE."""

import torch
from torch import nn
import torch.nn.functional as F


class RecurrentNet(nn.Module):

    """Recurrent Net used on text inputs and audio features."""

    def __init__(self, dict_args):
        """
        Initialize RecurrentNet.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(RecurrentNet, self).__init__()
        self.conv_outsize = dict_args["conv_outsize"]
        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]

        # lstm
        self.hidden = None
        self.init_hidden(self.batch_size)

        self.rnn = nn.GRUCell(
            self.word_embdim + self.conv_outsize, self.hidden_size)

        self.dropout_layer = nn.Dropout(self.dropout)

        # if self.attention:
            # self.attn_layer = Attention() # TODO: Build attention layer

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""
        if torch.cuda.is_available():
            self.hidden = torch.zeros(batch_size, self.hidden_size).cuda()
        else:
            self.hidden = torch.zeros(batch_size, self.hidden_size)

    def detach_hidden(self, batch_size):
        """Detach the hidden state of the RNN."""
        hidden_batch_size, _ = self.hidden.size()
        if hidden_batch_size != batch_size:
            self.init_hidden(batch_size)
        else:
            detached_hidden = self.hidden.detach()
            detached_hidden.zero_()
            self.hidden = detached_hidden

    def forward(self, seqembd, convfeatvects):
        """Forward pass."""
        # init output tensor
        seqlen, batch_size, _ = seqembd.size()
        log_probs = torch.zeros([seqlen, batch_size, self.vocab_size])
        if torch.cuda.is_available():
            log_probs = log_probs.cuda()

        for i in range(seqlen):
            i_t = seqembd[i]

            if self.attention:
                context = self.attn_layer(convfeatvects, self.hidden)
            else:
                context = convfeatvects

            # prep input
            i_t = torch.cat([i_t, context], dim=1)
            i_t = self.dropout_layer(i_t)

            # rnn forward
            self.hidden = self.rnn(i_t, self.hidden)
            output = self.hidden2vocab(self.hidden)
            log_probs[i] = F.log_softmax(output, dim=1)

        return log_probs, self.hidden
