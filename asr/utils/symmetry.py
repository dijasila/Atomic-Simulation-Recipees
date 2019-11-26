def is_symmetry_protected(kpt, op_scc):
    import numpy as np
    mirror_count = 0
    for symm in op_scc:
        # Inversion symmetry forces spin degeneracy and 180 degree rotation
        # forces the spins to lie in plane
        if (np.allclose(symm, -1 * np.eye(3)) or
            np.allclose(symm, np.array([-1, -1, -1] * np.eye(3)))):
            return True
        vals, vecs = np.linalg.eigh(symm)
        # A mirror plane
        if np.allclose(np.abs(vals), 1) and np.allclose(np.prod(vals), -1):
            # Mapping k -> k, modulo a lattice vector
            if np.allclose(kpt % 1, (np.dot(symm, kpt)) % 1):
                mirror_count += 1
    # If we have two or more mirror planes,
    # then we must have a spin-degenerate
    # subspace
    if mirror_count >= 2:
        return True
    return False
