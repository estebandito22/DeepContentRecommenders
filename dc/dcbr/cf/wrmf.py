"""Class for training a Recommendation Model using the Implicit library."""

import numpy as np
from implicit.als import AlternatingLeastSquares


class WRMF():

    """
    Class to train A Recommendation Model.

    Utilizes the implicit library.
    http://implicit.readthedocs.io/en/latest/index.html
    """

    def __init__(self, factors, l2, alpha, eps, n_iter):
        """
        Initialize the WRMF object.

        Args
            factors: The number of factors to use for the user and item
                embeddings.
            l2: [0,inf) the L2 regularization parameter.
            alpha: alpha paramter for score transformation.
            eps: eps paramter for score transformation.
            n_iter: the number of iterations to use for ALS optimization.
            train_pct: the percentage of data to use for training.  The
                remaining data is used for validation.
        """
        self.factors = factors
        self.l2 = l2
        self.alpha = alpha
        self.eps = eps
        self.n_iter = n_iter
        self.wrmf = AlternatingLeastSquares(
            factors=factors, regularization=l2,
            iterations=n_iter, calculate_training_loss=True)

    def conf_transform(self, item_user):
        """
        Transform raw playcounts to confidence scores.

        confidence = 1 + alpha * log(1 + playcount / eps)



        Args
            alpha: alpha parameter to use in the transformation.
            eps: eps parameter to use in the transformation.``

        Return
            item_user: csr matrix with items as rows, users as columns and
                confidence as values.
        -----------------------------------------------------------------------
        """
        item_user = item_user.tocoo()
        item_user.data = 1 + self.alpha*np.log1p(item_user.data/self.eps)

        return item_user.tocsr()

    def fit(self, item_user):
        """
        Train the WRMF model.

        Args
            item_user: an item_user matrix from CFDataHandler.
        """
        item_user = self.conf_transform(item_user)
        self.wrmf.fit(item_user)
