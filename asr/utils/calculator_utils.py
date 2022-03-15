"""Module contaning calculator utility functions.

The functions in this module basically only take a calculator as
essential arguments. Additional, non-essential arguments are allowed.

"""


def get_eigenvalues(calc, all_bz=False):
    """Get eigenvalues from calculator.

    Parameters
    ----------
    calc :   Calculator
    all_bz : return eigenvalues over all
             the Brillouin zone

    Returns
    -------
    e_skn: (ns, nk, nb)-shape array
    """
    import numpy as np
    rs = range(calc.get_number_of_spins())
    if all_bz:
        rk = calc.get_bz_to_ibz_map()
    else:
        rk = range(len(calc.get_ibz_k_points()))
    e = calc.get_eigenvalues
    return np.asarray([[e(spin=s, kpt=k) for k in rk] for s in rs])


def fermi_level(calc, eigenvalues=None, nelectrons=None, nspins=None):
    """Get Fermi level at T=0 from calculation.

    This works by filling in the appropriate number of electrons.

    Parameters
    ----------
    calc : ASE Calculator
        ASE calculator
    eigenvalues : ndarray, shape=(nspins, nkpoints, nbands)
        eigenvalues (taken from calc if None)
    nelectrons : float, optional
        number of electrons (taken from calc if None)
    nspins : int
        Number of spins that eigenvalues are provided for (default=2). Ie.
        2 when both spin-channels are represented in eps_skn or 1 if only
        1 spin-channel is represented in eps_skn.

    Returns
    -------
    fermi_level : float
        fermi level in eV
    """
    import numpy as np
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()

    if eigenvalues is None:
        eigenvalues = get_eigenvalues(calc)
        nspins = calc.get_number_of_spins()
    else:
        assert nspins is not None, 'You have to provide a number of spins!'

    eigenvalues_skn = eigenvalues  # More intuitive variable name
    eig_shape = eigenvalues_skn.shape

    assert len(eig_shape) == 3, f'Bad shape of eigenvalues: {eig_shape}.'
    nkpts = len(calc.get_bz_k_points())

    # The number of occupied states is the number of electrons
    # multiplied by the number of k-points
    nocc = int(nelectrons * nkpts)

    if eig_shape[1] == nkpts:
        count_k = np.ones(nkpts, int)
    else:
        weight_k = np.array(calc.get_k_point_weights())
        count_k = np.round(weight_k * nkpts).astype(int)

    if nspins == 1:
        count_k *= 2

    eps_N = np.repeat(eigenvalues_skn, count_k, axis=1).ravel()
    eps_N.sort()
    homo = eps_N[nocc - 1]
    lumo = eps_N[nocc]
    fermi_level = (homo + lumo) / 2
    return fermi_level
