import numpy as np
import scipy.signal as sp


def filter_boundary_matrix(hrfparam, X, condition='normal'):
    """
    This function attempts to optimize costly operations in the original code.

    :param fil_num:
    :param fil_den:
    :param X:
    :param condition:
    :return:
    """

    fil_num = hrfparam['num']
    fil_den = hrfparam['den']

    if condition == 'transpose':
        out = np.flip(sp.lfilter(fil_num, 1, np.flip(X, 0), axis=0), 0)
    else:
        out = sp.lfilter(fil_num, 1, X, axis=0)

    shiftnc = len(fil_den[1]) - 1
    if shiftnc > 0:
        out = np.vstack([np.zeros([shiftnc, X.shape[1]]), out, np.zeros([shiftnc, X.shape[1]])])

    if condition == 'normal':
        out = sp.lfilter([1], fil_den[0], out, axis=0)
        out = np.flip(sp.lfilter([1], fil_den[1], np.flip(out, axis=0), axis=0), axis=0) * fil_den[1][-1]
        if shiftnc > 0:
            out = out[2 * shiftnc + 1:]
    elif condition == 'transpose':
        out = np.flip(sp.lfilter([1], fil_den[0], np.flip(out, axis=0), axis=0), axis=0)
        out = sp.lfilter([1], fil_den[1], out, axis=0) * fil_den[1][-1]
        if shiftnc > 0:
            out = out[1:-2 * shiftnc]

    return out
