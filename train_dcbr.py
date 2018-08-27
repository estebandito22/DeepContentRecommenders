"""Script to train DCBR model."""

from argparse import ArgumentParser

from dc.deep_content import DCBR


def main(factors, l2, alpha, cf_eps, n_iter, n_splits, train_pct, n_recs,
         eval_pct, output_size, dropout, batch_size, lr, beta_one, beta_two,
         nn_eps, weight_decay, num_epochs, bn_momentum, data_type,
         triplets_txt, metadata_csv, save_dir, cf_only):
    """Train DCBR model."""
    dcbr = DCBR(factors, l2, alpha, cf_eps, n_iter, n_splits, train_pct,
                n_recs, eval_pct, output_size, dropout,
                batch_size, lr, beta_one, beta_two, nn_eps, weight_decay,
                num_epochs, bn_momentum, data_type)

    if cf_only:
        dcbr.fit_cf(triplets_txt)
    else:
        dcbr.fit(triplets_txt, metadata_csv, save_dir)


if __name__ == '__main__':
    """
    Usage:

    --factors 400 \
    --l2 10.0 \
    --alpha 4.0 \
    --cf_eps 1e-08 \
    --n_iter 16 \
    --n_splits 5 \
    --train_pct 0.75 \
    --n_recs 500 \
    --eval_pct 0.01 \
    --output_size 400 \
    --dropout 0 \
    --batch_size 32 \
    --lr 0.00001 \
    --weight_decay 0.0005 \
    --num_epochs 1000 \
    --bn_momentum 0.5 \
    --data_type scatter \
    --triplets_txt /scratch/swc419/DeepSong/data/MSD/train_triplets.txt \
    --metadata_csv /scratch/swc419/DeepSong/input/MSD/tracks.csv \
    --save_dir /scratch/swc419/DeepContentRecommenders/dcbr_models \
    --cf_only
    """
    ap = ArgumentParser()
    ap.add_argument("-f", "--factors", type=int,
                    help="Dimension of embedded factors.")
    ap.add_argument("-r", "--l2", type=float,
                    help="L2 regularization to use in CF model.")
    ap.add_argument("-a", "--alpha", type=float,
                    help="Alpha parameter to use in CF model.")
    ap.add_argument("-ce", "--cf_eps", type=float,
                    help="EPS parammeter to use in CF model.")
    ap.add_argument("-ni", "--n_iter", type=int,
                    help="Number of iterations to use in CF model.")
    ap.add_argument("-ns", "--n_splits", type=int,
                    help="Number of splits to use in CF model.")
    ap.add_argument("-tp", "--train_pct", type=float,
                    help="Percent of data for training to use in CF model.")
    ap.add_argument("-nr", "--n_recs", type=int,
                    help="Number of recommendations to use in CF model.")
    ap.add_argument("-ep", "--eval_pct", type=float,
                    help="Percent of users for evaluation to use in CF model.")
    ap.add_argument("-o", "--output_size", type=int,
                    help="Dimension of the target in NN model.  Corresponds \
                          to factors in CF model.")
    ap.add_argument("-d", "--dropout", type=float,
                    help="Percentage to use for dropout in NN model.")
    ap.add_argument("-bs", "--batch_size", type=int,
                    help="Number of samples per batch in NN model.")
    ap.add_argument("-lr", "--lr", type=float,
                    help="Learning rate to use in NN model.")
    ap.add_argument("-bo", "--beta_one", type=float, default=0.9,
                    help="Beta 1 parameter in NN model.")
    ap.add_argument("-bt", "--beta_two", type=float, default=0.999,
                    help="Beta 2 parameter in NN model.")
    ap.add_argument("-ne", "--nn_eps", type=float, default=1e-8,
                    help="EPS parameter in NN model.")
    ap.add_argument("-wd", "--weight_decay", type=float, default=0,
                    help="Weight decay parameter in NN model.")
    ap.add_argument("-e", "--num_epochs", type=int,
                    help="Number of epochs in NN model.")
    ap.add_argument("-bm", "--bn_momentum", type=float,
                    help="Momentum for Batch Normalization.")
    ap.add_argument("-dt", "--data_type",
                    help="'mel' or 'scatter' in NN model.")
    ap.add_argument("-tx", "--triplets_txt",
                    help="Path to triplets.txt file for training.")
    ap.add_argument("-mc", "--metadata_csv",
                    help="Path to metadata.csv file for training")
    ap.add_argument("-sd", "--save_dir",
                    help="Path to directory to save model.")
    ap.add_argument("-co", "--cf_only", action='store_true',
                    help="Only train CF model.")

    args = vars(ap.parse_args())

    main(args['factors'],
         args['l2'],
         args['alpha'],
         args['cf_eps'],
         args['n_iter'],
         args['n_splits'],
         args['train_pct'],
         args['n_recs'],
         args['eval_pct'],
         args['output_size'],
         args['dropout'],
         args['batch_size'],
         args['lr'],
         args['beta_one'],
         args['beta_two'],
         args['nn_eps'],
         args['weight_decay'],
         args['num_epochs'],
         args['bn_momentum'],
         args['data_type'],
         args['triplets_txt'],
         args['metadata_csv'],
         args['save_dir'],
         args['cf_only'])
