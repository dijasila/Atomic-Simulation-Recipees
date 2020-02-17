import numpy as np
from types import SimpleNamespace


class DielectricFunction:

    chi0 = SimpleNamespace(plasmafreq_vv=np.zeros((3, 3), float))

    def __init__(self, *args, **kwargs):
        pass

    def get_frequencies(self):
        return np.linspace(0, 10, 100)

    def get_polarizability(self, *args, **kwargs):
        alpha_w = np.zeros((100, ), float) + 1j * 0
        return alpha_w, alpha_w
