"""Script to train DCBR model."""

from argparse import ArgumentParser
import numpy as np

from dc.nn.dcuelm import DCUELM


def main(feature_dim, batch_size, neg_batch_size, u_embdim, margin, lr,
         beta_one, beta_two, eps, weight_decay, num_epochs, bn_momentum,
         dropout, model_type, data_type, n_users, n_items, eval_pct,
         word_embdim, word_embeddings, hidden_size, dropout_rnn, vocab_size,
         attention, triplets_txt, metadata_csv, save_dir):
    """Train DCUELM model."""
    if word_embeddings is not None:
        word_embeddings = np.loadtxt(word_embeddings)

    dcue = DCUELM(feature_dim, batch_size, neg_batch_size, u_embdim, margin,
                  lr, beta_one, beta_two, eps, weight_decay, num_epochs,
                  bn_momentum, dropout, model_type, data_type, n_users,
                  n_items, eval_pct, word_embdim, word_embeddings, hidden_size,
                  dropout_rnn, vocab_size, attention)

    dcue.fit(triplets_txt, metadata_csv, save_dir)


if __name__ == '__main__':
    """
    Usage:

    python train_dcue.py \
    --feature_dim 100 \
    --batch_size 64 \
    --neg_batch_size 20 \
    --u_embdim 300 \
    --margin 0.2 \
    --learning_rate 0.00001 \
    --weight_decay 0.0 \
    --num_epochs 1000 \
    --bn_momentum 0.5 \
    --data_type mel \
    --n_users 20000 \
    --n_items 10000 \
    --triplets_txt /scratch/swc419/DeepSong/data/MSD/train_triplets.txt \
    --metadata_csv /scratch/swc419/DeepSong/input/MSD/tracks.csv \
    --save_dir dcue_models
    """

    ap = ArgumentParser()
    ap.add_argument("-f", "--feature_dim", type=int,
                    help="Dimension of the feature embedding vector, \
                          corresponds to dimension of \
                          collaborative filtering latent factors")
    ap.add_argument("-b", "--batch_size", type=int, default=32,
                    help="Batch size to use for training the Conv")
    ap.add_argument("-nb", "--neg_batch_size", type=int, default=32,
                    help="Batch size to use for negative sampling training \
                          the Conv")
    ap.add_argument("-u", "--u_embdim", type=int,
                    help="Dimension of the user embedding")
    ap.add_argument("-m", "--margin", type=float,
                    help="Margin to use in hinge loss")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="Learning rate to use in ADAM")
    ap.add_argument("-b1", "--beta_one", type=float, default=0.9,
                    help="Beta 1 to use in ADAM")
    ap.add_argument("-b2", "--beta_two", type=float, default=0.999,
                    help="Beta 2 to use in ADAM")
    ap.add_argument("-ep", "--eps", type=float, default=1e-8,
                    help="EPS to use in ADAM")
    ap.add_argument("-wd", "--weight_decay", type=float, default=0,
                    help="Weight decay to use in ADAM")
    ap.add_argument("-ne", "--num_epochs", type=int, default=50,
                    help="Number of epochs to train")
    ap.add_argument("-bm", "--bn_momentum", type=float,
                    help="Momentum for Batch Normalization.")
    ap.add_argument("-do", "--dropout", type=float,
                    help="Dropout to use in audiomodel.")
    ap.add_argument("-mt", "--model_type",
                    help="scatter or mel or mel2")
    ap.add_argument("-dt", "--data_type",
                    help="scatter or mel")
    ap.add_argument("-nu", "--n_users", type=int, default=20000,
                    help="Number of users to include.")
    ap.add_argument("-ni", "--n_items", type=int, default=10000,
                    help="Number of items to include.")
    ap.add_argument("-p", "--eval_pct", type=float, default=0.025,
                    help="Number of items to include.")
    ap.add_argument("-wed", "--word_embdim", type=int, default=300,
                    help="Embedding dimension for word vectors.")
    ap.add_argument("-we", "--word_embeddings", default=None,
                    help="Path to pretrained word embeddings.")
    ap.add_argument("-hs", "--hidden_size", type=int, default=512,
                    help="RNN hidden dimension.")
    ap.add_argument("-dor", "--dropout_rnn", type=float, default=0,
                    help="Dropout percentage to use in RNN.")
    ap.add_argument("-vs", "--vocab_size", type=int, default=20000,
                    help="Vocab size of the word embedding.")
    ap.add_argument("-at", "--attention", action='store_true',
                    help="Use attention in RNN.")
    ap.add_argument("-tx", "--triplets_txt",
                    help="Path to triplets.txt file for training.")
    ap.add_argument("-mc", "--metadata_csv",
                    help="Path to metadata.csv file for training")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model")

    args = vars(ap.parse_args())

    main(args['feature_dim'],
         args['batch_size'],
         args['neg_batch_size'],
         args['u_embdim'],
         args['margin'],
         args['learning_rate'],
         args['beta_one'],
         args['beta_two'],
         args['eps'],
         args['weight_decay'],
         args['num_epochs'],
         args['bn_momentum'],
         args['dropout'],
         args['model_type'],
         args['data_type'],
         args['n_users'],
         args['n_items'],
         args['eval_pct'],
         args['word_embdim'],
         args['word_embeddings'],
         args['hidden_size'],
         args['dropout_rnn'],
         args['vocab_size'],
         args['attention'],
         args['triplets_txt'],
         args['metadata_csv'],
         args['save_dir'])
