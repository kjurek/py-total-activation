from os.path import join

import numpy as np
import os
import scipy.io as sio

from TestBase import TestBase
from TestBase import traverse_matlab_files
from TotalActivation.process.temporal import temporal_TA
from TotalActivation import TotalActivation


class MatlabComparisonTestTemporalTA(TestBase):
    """
    def get(self, in_, field, default=None):
        try:
            return in_[field].flatten()
        except ValueError:
            return default

    def get_input(self, m):
        y = m['y']
        f_Analyze = self.get(m, 'f_Analyze')
        f_Analyze = {'num': f_Analyze['num'][0][0],
                     'den': np.array([f_Analyze['den'][0][0][0].flatten(),
                                      f_Analyze['den'][0][0][1].flatten()])}
        MaxEig = self.get(m, 'MaxEig')[0]
        Dimension = self.get(m, 'Dimension')[-1]
        NitTemp = self.get(m, 'NitTemp')[0]
        NoiseEstimateFin = self.get(m, 'NoiseEstimateFin')
        VxlInd = self.get(m, 'VxlInd')[0]
        LambdaTemp = self.get(m, 'LambdaTemp')
        COST_SAVE = self.get(m, 'COST_SAVE')

        return y, f_Analyze, MaxEig, Dimension, NitTemp, NoiseEstimateFin, VxlInd, LambdaTemp, COST_SAVE

    def get_expected_output(self, m):
        x = m['x']
        NoiseEstimateFin = self.get(m, 'NoiseEstimateFin')
        LambdasTempFin = self.get(m, 'LambdasTempFin')
        CostTemp = self.get(m, 'CostTemp')

        return x, NoiseEstimateFin, LambdasTempFin, CostTemp

    def test_TemporalTA(self):
        for input_file in traverse_matlab_files(join(self.data_path, 'TemporalTA')):
            m = sio.loadmat(input_file)
            x, NoiseEstimateFin, LambdasTempFin, CostTemp = temporal_TA(*self.get_input(m))
            e_x, e_NoiseEstimateFin, e_LambdasTempFin, e_costTemp = self.get_expected_output(m)

            np.testing.assert_almost_equal(x, e_x)
            np.testing.assert_almost_equal(NoiseEstimateFin, e_NoiseEstimateFin)
            np.testing.assert_almost_equal(LambdasTempFin, e_LambdasTempFin)
            np.testing.assert_almost_equal(CostTemp.flatten(), e_costTemp)
    """

    def test_TemporalTA(self):
        test_ta = TotalActivation()
        test_ta.load_text_data(os.path.join(self.data_path, 'TemporalTA/input_matlab_normalized.csv'),)
        test_ta.data = test_ta.data[:,0:10]
        test_ta.n_voxels = 10
        test_ta.config['TR'] = 1
        test_ta._get_hrf_parameters()
        test_ta.t_iter = 2500
        test_ta._temporal()

        expected_output = np.genfromtxt(os.path.join(self.data_path, 'TemporalTA/phantom_block_nospatial.csv'),
                                        delimiter=',')[:,0:10]

        np.testing.assert_almost_equal(test_ta.deconvolved_, expected_output, decimal = 3)