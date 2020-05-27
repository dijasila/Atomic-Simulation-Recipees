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
    from gpaw.occupations import occupation_numbers
    from ase.units import Ha
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()
    if eps_skn is None:
        eps_skn = eigenvalues(calc)
    eps_skn.sort(axis=-1)
    occ = calc.occupations.todict()
    if width is not None:
        occ['width'] = width
    weight_k = calc.get_k_point_weights()
    return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha
