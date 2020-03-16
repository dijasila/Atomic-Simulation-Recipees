def atoms2symmetry(atoms):
    from types import SimpleNamespace
    import numpy as np

    return SimpleNamespace(op_scc=[np.eye(3)],
                           has_inversion=False)
