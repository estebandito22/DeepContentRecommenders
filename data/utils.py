"""Method to rate limit API calls."""

import time
import threading

from functools import wraps
from scipy.sparse import csr_matrix


def rate_limited(max_per_second):
    """
    Rate limit function calls.

    Courtesy of https://gist.github.com/gregburek/1441055
    """
    lock = threading.Lock()
    min_interval = 1.0 / float(max_per_second)

    def decorate(func):
        """Decorate function."""
        last_time_called = [0.0]

        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            """Rate limit function."""
            lock.acquire()
            elapsed = time.clock() - last_time_called[0]
            left_to_wait = min_interval - elapsed

            if left_to_wait > 0:
                time.sleep(left_to_wait)

            lock.release()

            ret = func(*args, **kwargs)
            last_time_called[0] = time.clock()
            return ret

        return rate_limited_function

    return decorate


def item_user_matrix(X):
    """
    Build the csr matrix of item x user.

    Args
        X: Dataframe with 'item', 'user', 'score' fields.

    Return
        item_user: csr matrix with items as rows, users as columns and score
                   as values.
        item_index: dictionary of the item indexes in the item_user matrix
                    for each item id.
        user_index: dicationary of the user indexes in the item_user matrix
                    for each user_id.

    ---------------------------------------------------------------------------
    """
    X['user_id'] = X['user_id'].astype("category")
    X['song_id'] = X['song_id'].astype("category")

    row = X['song_id'].cat.codes.copy()
    col = X['user_id'].cat.codes.copy()

    nrow = len(X['song_id'].cat.categories)
    ncol = len(X['user_id'].cat.categories)

    item_user = csr_matrix((X['score'], (row, col)), shape=(nrow, ncol))

    user = dict(enumerate(X['user_id'].cat.categories))
    user_index = {u: i for i, u in user.items()}

    item = dict(enumerate(X['song_id'].cat.categories))
    item_index = {s: i for i, s in item.items()}

    return item_user, item_index, user_index
