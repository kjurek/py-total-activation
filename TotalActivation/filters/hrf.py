import numpy as np
from cons import cons_filter

class BoldParameters(object):
    def __init__(self):
        a1 = -1.020408163265306
        a2 = -3.092145949288806
        a3 = -0.324675324675325 - 0.548716683350910j
        a4 = -0.324675324675325 + 0.548716683350910j
        self.a = np.ndarray(a1, a2, a3, a4)
        self.psi = -11.898107445013801

class SpmhrfParameters(object):
    def __init__(self):
        a1 = -0.27
        a2 = -0.27
        a3 = -0.4347 - 1j * 0.3497
        a4 = -0.4347 + 1j * 0.3497
        self.a = np.ndarray(a1, a2, a3, a4)
        self.psi = -0.1336

class HrfFilter(object):
    def __init__(self, hrf_parameters, t_r):
        self.fil_zeros = hrf_parameters.a * t_r
        self.fil_poles = hrf_parameters.psi * t_r
        cons = 1
        hnum = cons_filter(self.fil_zeros) * cons
        hden = cons_filter(self.fil_poles)
