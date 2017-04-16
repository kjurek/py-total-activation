from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pywt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def mad(X, axis=0):
    """
    Median absolute deviation

    :param X:
    :return:
    """

    y = np.median(np.abs(X - np.median(X, axis=axis)), axis=axis)
    return y


def wiener(X, hrfparam, config):
    """

    :param X:
    :param hrfparam:
    :param config:
    :return:
    """


    f_num = np.abs(np.fft.fft(hrfparam[0]['num'], 200) ** 2)

    f_den = np.abs(np.fft.fft(hrfparam[0]['den'][0], 200) * \
                   np.fft.fft(hrfparam[0]['den'][1], 200) * \
                   t.hrfparams[0]['den'][-1] * \
                   np.exp(np.arange(1, 201) * (t.hrfparams[0]['den'][1].shape[0] - 1) / 200)) ** 2

    _, coef = pywt.wavedec(X, 'db3', level=1, axis=0)
    lambda_temp = mad(coef) * config['Lambda'] ** 2 * 200

    res = np.real(np.fft.ifft(np.fft.fft(X) * (np.repeat(f_den, 10).reshape(200, 10) / (
        np.repeat(f_den, 10).reshape(200, 10) + np.kron(f_num, lambda_temp).reshape(200, 10))), axis=1))

    return res

