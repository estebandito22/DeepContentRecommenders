"""PyTorch class for word embedding in DCUELM model."""

import torch.nn as nn


class WordEmbeddings(nn.Module):

    """Class to embed words."""

    def __init__(self, dict_args):
        """
        Initialize WordEmbeddings.

        Args
            dict_args: dictionary containing the following keys:
                word_embdim: The dimension of the lookup embedding.
                vocab_size: The count of words in the data set.
                word_embeddings: Pretrained embeddings.
        """
        super(WordEmbeddings, self).__init__()

        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.word_embeddings = dict_args["word_embeddings"]

        self.embeddings = nn.Embedding(self.vocab_size, self.word_embdim)

        if self.word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(self.word_embeddings)
            self.embeddings.weight.requires_grad = False

    def forward(self, seq):
        """
        Forward pass.

        Args
            seq: A tensor of sequences of word indexes of size
                 batch_size x seqlen.
        """
        # seq: batch_size x seqlen
        return self.embeddings(seq)  # batch_size x seqlen x embd_dim
