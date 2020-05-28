import numpy as np


class Symmetry:

    def __init__(self, *args, **kwargs):
        self.op_scc = [np.eye(3)]
        self.has_inversion = False

    def analyze(self, *args, **kwargs):
        pass
