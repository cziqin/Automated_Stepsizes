import scipy
import numpy as np

import numpy.linalg as la
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import expit


def safe_sparse_add(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # on of them is non-sparse, convert
        # everything to dense.
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


def logsig(x):
    """
    Compute the log-sigmoid function component-wise.
    See http://fa.bianp.net/blog/2019/evaluate_logistic/ for more details.
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def logistic_loss(params, samples, targets, l2):
    z = np.dot(samples, params)
    y = np.asarray(targets)
    return np.mean((1 - y) * z - logsig(z)) + (l2 / 2) * la.norm(params) ** 2


# def logistic_loss(params, samples, targets, l2):
#     z = np.dot(samples, params)
#     y = np.asarray(targets)
#     sigmoid_z = expit(z)
#     log_loss = -np.mean(targets * np.log(sigmoid_z) + (1 - targets) * np.log(1 - sigmoid_z))
#     l2_penalty = l2 / 2 * np.linalg.norm(params) ** 2
#     total_loss = log_loss + l2_penalty
#     return total_loss


def logistic_gradient(params, samples, targets, l2, normalize=True):
    """
    Gradient of the logistic loss at point w with features X, labels y and l2 regularization.
    If labels are from {-1, 1}, they will be changed to {0, 1} internally
    """
    y = (targets + 1) / 2 if -1 in targets else targets
    activation = scipy.special.expit(safe_sparse_dot(samples, params, dense_output=True).ravel())
    grad = safe_sparse_add(samples.T.dot(activation - y) / samples.shape[0], l2 * params)
    grad = np.asarray(grad).ravel()

    if normalize:
        return grad
    return grad * len(y)


# def logistic_gradient(params, samples, targets, l2, normalize=True):
#     """
#     Gradient of the logistic loss at point w with features X, labels y and l2 regularization.
#     If labels are from {-1, 1}, they will be changed to {0, 1} internally
#     """
#     y = (targets + 1) / 2 if -1 in targets else targets
#
#     # Compute the predictions
#     z = np.dot(samples, params)
#     sigmoid_z = expit(z)  # Sigmoid activation
#
#     # Calculate gradient
#     error = sigmoid_z - y
#     gradient = np.dot(samples.T, error) / len(samples)
#
#     # Add L2 regularization
#     gradient += l2 * params
#
#     # Normalize gradient if required
#     if normalize:
#         gradient /= len(samples)
#
#     return gradient