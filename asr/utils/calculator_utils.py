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


def fermi_level(calc, eps_skn=None, nelectrons=None,
                width=None):
    """Get Fermi level from calculation.

    Parameters
    ----------
    calc : GPAW
        GPAW calculator
    eps_skn : ndarray, shape=(ns, nk, nb), optional
        eigenvalues (taken from calc if None)
    nelectrons : float, optional
        number of electrons (taken from calc if None)
    width : float, optional
        Fermi dirac width, if None then inherit from calc

    Returns
    -------
    out : float
        fermi level
    """
    import numpy as np
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()
    nkpts = calc.get_number_of_kpts()
    if eps_skn is None:
        eps_skn = eigenvalues(calc)
    count_k = np.round(calc.get_k_point_weights() * nkpts)
    eps_N = np.repeat(eps_skn, count_k, axis=1).ravel()
    homo = eps_N[nelectrons]
    lumo = eps_N[nelectrons + 1]
    fermi_level = (homo + lumo) / 2
    return fermi_level
