import numpy as np

EPSILON = 1e-10


def numpy_softmax(x, axis=None):
    m = np.max(x)
    _x = x - m
    _exp = np.exp(_x)
    return _exp / (np.sum(_exp, keepdims=True, axis=axis) + EPSILON)


def top_1_accuracy_from_pred(p, q):
    _p = np.atleast_2d(p)
    _q = np.atleast_2d(q)
    acc = np.mean(np.argmax(_p, axis=-1) == np.argmax(_q, axis=-1))
    return acc


def softmax_xent_from_pred(p, q):
    assert np.isclose(p.sum(), 1.) and np.isclose(
        q.sum(), 1.), 'Distributions Provided Invalid'
    _p = np.atleast_2d(p)
    _q = np.atleast_2d(q)
    xent = np.sum(_p * (np.log(_q + EPSILON)), axis=-1)
    return np.mean(xent)
