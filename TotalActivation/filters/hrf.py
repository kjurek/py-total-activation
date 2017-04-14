import numpy as np
from scipy import signal

from TotalActivation.filters.cons import cons


class BoldParameters(object):
    def __init__(self):
        eps = 0.54
        ts = 1.54
        tf = 2.46
        t0 = 0.98
        alpha = 0.33
        E0 = 0.34
        V0 = 1
        k1 = 7 * E0
        k2 = 2
        k3 = 2 * E0 - 0.2
        c = (1 + (1 - E0) * np.log(1 - E0) / E0) / t0

        a1 = -1 / t0
        a2 = -1 / (alpha * t0)
        a3 = -(1 + 1j * np.sqrt(4 * np.power(ts, 2) / tf - 1)) / (2 * ts)
        a4 = -(1 - 1j * np.sqrt(4 * np.power(ts, 2) / tf - 1)) / (2 * ts)

        self.psi = -((k1 + k2) * ((1 - alpha) / alpha / t0 - c / alpha) - (k3 - k2) / t0) / (-(k1 + k2) * c * t0 - k3 + k2)
        self.a = np.array([a1, a2, a3, a4])


class SpmhrfParameters(object):
    def __init__(self):
        a1 = -0.27
        a2 = -0.27
        a3 = -0.4347 - 1j * 0.3497
        a4 = -0.4347 + 1j * 0.3497
        self.a = np.array([a1, a2, a3, a4])
        self.psi = -0.1336


class HrfFilter(object):
    def __init__(self, hrf_parameters, t_r):
        self.t_r = t_r
        self.fil_zeros = hrf_parameters.a * t_r
        self.fil_poles = np.array([hrf_parameters.psi * t_r])
        self.hnum = cons(self.fil_zeros)
        self.hden = cons(self.fil_poles)
        self.causal = np.array([x for x in self.fil_poles if np.real(x) < 0])
        self.n_causal = np.array([x for x in self.fil_poles if np.real(x) > 0])
        self.h_dc = cons(self.causal)
        self.h_dnc = cons(self.n_causal)
        self.reconstruct = {'num': self.hnum, 'den': np.array([self.h_dc, self.h_dnc])}


class Spike(HrfFilter):
    def __init__(self, hrf_parameters, t_r):
        super(Spike, self).__init__(hrf_parameters, t_r)

    def compute(self):
        d2, d1 = signal.freqz(self.hnum, self.hden, 1024)
        maxeig = np.max(np.power(np.abs(d1), 2))
        return self.reconstruct, self.reconstruct, maxeig


class Block(HrfFilter):
    def __init__(self, hrf_parameters, t_r):
        super(Block, self).__init__(hrf_parameters, t_r)

    def compute(self):
        self.fil_zeros = np.append(self.fil_zeros, 0)
        hnum2 = cons(self.fil_zeros)
        d2, d1 = signal.freqz(hnum2, self.hden, 1024)
        maxeig = np.max(np.power(np.abs(d1), 2))

        return {'num': hnum2, 'den': self.reconstruct['den']}, self.reconstruct, maxeig
