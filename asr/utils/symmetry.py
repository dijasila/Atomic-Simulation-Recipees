def restrict_spin_projection_2d(kpt, op_scc, s_vm):
    import numpy as np
    mirror_count = 0
    s_vm = s_vm.copy()
    for symm in op_scc:
        # Inversion symmetry forces spin degeneracy and 180 degree rotation
        # forces the spins to lie in plane
        if np.allclose(symm, -1 * np.eye(3)):
            s_vm[:] = 0
            continue
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
        s_vm[2, :] = 0

    return s_vm
