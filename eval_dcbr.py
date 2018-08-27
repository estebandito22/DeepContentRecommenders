"""Script to evaluate DCBR model."""

from argparse import ArgumentParser

import numpy as np

from dc.deep_content import DCBR


def main(model_dir, epoch, eval_pct):
    """Eval DCBR model."""
    # load model
    dcbr = DCBR()
    dcbr.load(model_dir, epoch)

    # generate random user list for evaluation
    users = list(dcbr.dh.user_index.keys())
    n_users = len(users)
    users_sample = np.random.choice(users, int(n_users * eval_pct))

    # evaluate cf model
    cf_score = dcbr.score(users_sample, 'test')

    # evaluate nn model
    dcbr.insert_nn_factors()
    nn_score = dcbr.score(users_sample, 'test')

    print("CF Model AUC: {}\tNN Model AUC: {}".format(cf_score, nn_score))


if __name__ == '__main__':
    """
    Usage:

    --model_dir /scratch/swc419/DeepContentRecommenders/cf_models/item_400_reg_10.0_iter_1_alpha_4.0_eps_1e-08_CONV_drop_0.0_lr_3.532465151309649e-05_b1_0.9_b2_0.999_wd_3.0462089048836577e-05 \
    --epoch 10 \
    --eval_pct 0.01
    """
    ap = ArgumentParser()
    ap.add_argument("-d", "--model_dir",
                    help="Directory where model is saved.")
    ap.add_argument("-e", "--epoch", type=int,
                    help="Epoch to load.")
    ap.add_argument("-p", "--eval_pct", type=float, default=0.01,
                    help="(0,1] percent of users to evaluate.")
    args = vars(ap.parse_args())

    main(args['model_dir'],
         args['epoch'],
         args['eval_pct'])
