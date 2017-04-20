import pywt
import numpy as np

from temporal import temporal_TA

def mad(X, axis=0):
    """
    Median absolute deviation

    :param X: Input matrix
    ;param axis: Axis to calculate quantity (default = 0)
    :return: MAD for X along axis
    """

    return np.median(np.abs(X - np.median(X, axis=axis)), axis=axis)

def parallel_temporalTA(input, output, voxels, l, f_Analyze, maxeig, n_tp, t_iter, cost_save):
    """
    This function allows to run a process and dump the results to memory-shared object

    :param input:
    :param output:
    :param voxels:
    :return:
    """

    output[:, voxels] = temporal_TA(input, f_Analyze, maxeig, n_tp, t_iter,
                                    noise_estimate_fin=None, l = l, cost_save=cost_save,
                                    voxels=voxels)