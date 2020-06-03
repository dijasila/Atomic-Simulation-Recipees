"""Module contaning calculator utility functions.

The functions in this module basically only take a calculator as
essential arguments. Additional, non-essential arguments are allowed.

"""


def eigenvalues(calc):
    """Get eigenvalues from calculator.

    Parameters
    ----------
    calc : Calculator

    Returns
    -------
    e_skn: (ns, nk, nb)-shape array
    """
    import numpy as np
    rs = range(calc.get_number_of_spins())
    rk = range(len(calc.get_ibz_k_points()))
    e = calc.get_eigenvalues
    return np.asarray([[e(spin=s, kpt=k) for k in rk] for s in rs])


def fermi_level(calc, eps_skn=None, nelectrons=None):
    """Get Fermi level at T=0 from calculation.

    This works by filling in the appropriate number of electrons.

    Parameters
    ----------
    calc : GPAW
        GPAW calculator
    eps_skn : ndarray, shape=(ns, nk, nb), optional
        eigenvalues (taken from calc if None)
    nelectrons : float, optional
        number of electrons (taken from calc if None)

    Returns
    -------
    fermi_level : float
        fermi level in eV
    """
    import numpy as np
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()

    nkpts = len(calc.get_bz_k_points())

    # The number of occupied states is the number of electrons
    # multiplied by the number of k-points
    nocc = int(nelectrons * nkpts)
    if eps_skn is None:
        eps_skn = eigenvalues(calc)
    weight_k = np.array(calc.get_k_point_weights())
    count_k = np.round(weight_k * nkpts).astype(int)
    eps_N = np.repeat(eps_skn, count_k, axis=1).ravel()
    eps_N.sort()
    homo = eps_N[nocc - 1]
    lumo = eps_N[nocc]
    fermi_level = (homo + lumo) / 2
    return fermi_level
