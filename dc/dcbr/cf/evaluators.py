"""Evaluation functions for cf model."""

from collections import defaultdict

import pandas as pd

from dc.dcbr.cf.evaluation import train_test_split
from dc.dcbr.cf.evaluation import precision_at_k
from dc.dcbr.cf.evaluation import mean_average_precision_at_k


def cross_validate(dh, estimator, n_splits=4, train_pct=0.75, n_recs=500,
                   eval_pct=0.15):
    """
    Perform random shuffle cross validation.

    Args
        dh: an initialized CFDataHandler
        estimator: WRMF estimator
        n_splits: The number of splits to use for cross validation.
        train_pct: The percentage of data to use in the training fold.
        n_recs: The number of recommendations to use for evaluation.
        eval_pct: The percent of users to use for evaluation.

    Return
        Dataframe of mAP and Precision at K for both the train and the test
        folds of each cross validation iteration.

    ---------------------------------------------------------------------------
    """
    summary = defaultdict(list)

    for i in range(n_splits):

        print('Training split {}'.format(i))

        item_user_train, item_user_test = dh.train_val_split(train_pct)

        estimator.fit(item_user_train)
        user_item_train_sample, user_item_test_sample = dh.sample_users(
            item_user_train, item_user_test, eval_pct)

        p_at_k_test = precision_at_k(
            estimator.wrmf, user_item_train_sample, user_item_test_sample,
            K=n_recs, show_progress=True, num_threads=0)

        p_at_k_train = precision_at_k(
            estimator.wrmf, user_item_test_sample, user_item_train_sample,
            K=n_recs, show_progress=True, num_threads=0)

        map_at_k_test = mean_average_precision_at_k(
            estimator.wrmf, user_item_train_sample, user_item_test_sample,
            K=n_recs, show_progress=True, num_threads=0)

        map_at_k_train = mean_average_precision_at_k(
            estimator.wrmf, user_item_test_sample, user_item_train_sample,
            K=n_recs, show_progress=True, num_threads=0)

        # Record statistics for the train/test split
        summary['test_p_at_k'] += [p_at_k_test]
        summary['train_p_at_k'] += [p_at_k_train]
        summary['test_map_at_k'] += [map_at_k_test]
        summary['train_map_at_k'] += [map_at_k_train]

    return pd.DataFrame(summary)


def evaluate(dh, estimator, n_splits=5, n_recs=500, eval_pct=0.15,
             excluded_item_indexes=None):
    """
    Perform evaluation on the test set.

    Arg
        item_user_train: CSR matrix with items along rows, users in columns and
                         confidence scores as values.
        item_user_test: CSR matrix with items along rows, users in columns and
                        confidence scores as values.
        estimator: WRMF estimator
        n_recs: The number of recommendations to use for evaluation.
        eval_pct: pct of users to use for evaluation
        excluded_item_indexes: list of items indexes to exclude

    Return
        Dataframe of mAP and Precision at K for both the train and the test
        folds of each cross validation iteration.

    ---------------------------------------------------------------------------
    """
    summary = defaultdict(list)

    item_index_complement = dh.item_complement(excluded_item_indexes)

    for _ in range(n_splits):

        user_item_train_sample, user_item_test_sample = dh.sample_users(
            eval_pct)

        p_at_k_test = precision_at_k(
            estimator.wrmf, user_item_train_sample, user_item_test_sample,
            K=n_recs, show_progress=True, num_threads=0)

        p_at_k_test_filt = precision_at_k(
            estimator.wrmf, user_item_train_sample, user_item_test_sample,
            filter_items=item_index_complement, K=n_recs,
            show_progress=True, num_threads=0)

        map_at_k_test = mean_average_precision_at_k(
            estimator.wrmf, user_item_train_sample, user_item_test_sample,
            K=n_recs, show_progress=True, num_threads=0)

        map_at_k_test_filt = mean_average_precision_at_k(
            estimator.wrmf, user_item_train_sample, user_item_test_sample,
            filter_items=item_index_complement, K=n_recs,
            show_progress=True, num_threads=0)

        # Record statistics for the train/test split
        summary['test_p_at_k'] += [p_at_k_test]
        summary['test_filt_p_at_k'] += [p_at_k_test_filt]
        summary['test_map_at_k'] += [map_at_k_test]
        summary['test_filt_map_at_k'] += [map_at_k_test_filt]

    return pd.DataFrame(summary)
