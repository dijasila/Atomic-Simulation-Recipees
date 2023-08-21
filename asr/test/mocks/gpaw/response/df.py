import numpy as np
from types import SimpleNamespace


class DielectricFunction:

    def __init__(self, *args, **kwargs):
        self.chi0calc = Chi0Calculator()

    def get_frequencies(self):
        return np.linspace(0, 10, 100)

    def get_polarizability(self, *args, **kwargs):
        alpha_w = np.zeros((100, ), float) + 1j * 0
        alpha_w[10] = 1 + 1j
        return alpha_w, alpha_w


class Chi0Calculator:

    def __init__(self, *args, **kwargs):
        self.chi0_opt_ext_calc = Chi0OpticalExtensioneCalculator()

    def check_high_symmetry_ibz_kpts(self):
        pass


class Chi0OpticalExtensioneCalculator:
    def __init__(self, wd=0, rate=0):
        self.wd = wd
        self.rate = rate
        self.drude_calc = Chi0DrudeCalculator()


class Chi0DrudeCalculator:

    def __init__(self, *args, **kwargs):
        pass

    def calculate(self, *args, **kwargs):
        chi0_drude = SimpleNamespace(plasmafreq_vv=np.zeros((3, 3), float))
        return chi0_drude
