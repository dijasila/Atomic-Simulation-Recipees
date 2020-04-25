def get_number_of_electrons(angular):
    """Compute the number of electrons to fill orbitals."""
    count = 0
    count += 2 * angular.count('s')
    count += 6 * angular.count('p')
    count += 10 * angular.count('d')
    count += 14 * angular.count('f')

    return count


def raw_orbital_LDOS(calc, a, spin, angular='spdf', nbands=None, **kwargs):
    """Distribute weights on atoms, spins and angular momentum."""
    import numpy as np
    from ase.units import Ha
    from asr.pdos import get_l_a

    # Extract eigenvalues and weights
    e_kn = calc.get_all_eigenvalues() / Ha
    w_k = np.array(calc.get_k_point_weights())

    # Take care of spin degeneracy
    if calc.get_number_of_spins() == 1:
        w_k *= 2

    # Figure out the total number of orbitals to be counted
    l_a = get_l_a(calc.atoms.get_atomic_numbers())
    nstates = 0
    for a, orbitals in l_a.items():
        nstates += get_number_of_electrons(orbitals)
    # Normalize weights
    w_k *= get_number_of_electrons(angular) / nstates

    # Return flattened energies and weights
    nb = e_kn.shape[1]
    assert nbands is None or nbands <= nb
    nb = nbands if nbands is not None else nb

    return e_kn[:, :nb].flatten(), np.tile(w_k, nb)
