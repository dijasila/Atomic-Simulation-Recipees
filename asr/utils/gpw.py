"""Utility functions to read .gpw files"""

import numpy as np

from ase.units import Ha
from ase.parallel import broadcast

from gpaw import GPAW
from gpaw import mpi
from gpaw.spinorbit import get_spinorbit_eigenvalues
from gpaw.occupations import occupation_numbers


def gpw2eigs(gpw, soc=True, bands=None, return_spin=False,
             optimal_spin_direction=False):
    """Get the eigenvalues w or w/o spinorbit coupling and the corresponding
    Fermi energy from a gpaw calculator

    Parameters:
    -----------
    gpw : str
        gpw filename
    soc : None, bool
        Use spinorbit coupling. If None it returns both w and w/o
    optimal_spin_direction : bool
        If True, use get_spin_direction to calculate the spin direction
        for the SOC
    bands: slice, list of ints or None
        None gives parameters.convergence.bands if possible else all bands

    Returns:
    --------
    dict or e_skn, efermi
        containg eigenvalues and fermi levels w and w/o spinorbit coupling
    """
    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    dct = None
    if mpi.world.rank in ranks:
        theta = 0
        phi = 0
        if optimal_spin_direction:
            theta, phi = get_spin_direction()
        calc = GPAW(gpw, txt=None, communicator=comm)
        if bands is None:
            n2 = calc.todict().get('convergence', {}).get('bands')
            bands = slice(0, n2)
        if isinstance(bands, slice):
            bands = range(calc.get_number_of_bands())[bands]
        eps_nosoc_skn = eigenvalues(calc)[..., bands]
        efermi_nosoc = calc.get_fermi_level()
        eps_mk, s_kvm = get_spinorbit_eigenvalues(calc, bands=bands,
                                                  theta=theta,
                                                  phi=phi,
                                                  return_spin=True)
        eps_km = eps_mk.T
        efermi = fermi_level(calc, eps_km[np.newaxis],
                             nelectrons=2 *
                             calc.get_number_of_electrons())
        dct = {'eps_nosoc_skn': eps_nosoc_skn,
               'eps_km': eps_km,
               'efermi_nosoc': efermi_nosoc,
               'efermi': efermi,
               's_kvm': s_kvm}

    dct = broadcast(dct, root=0, comm=mpi.world)
    if soc is None:
        return dct
    elif soc:
        out = (dct['eps_km'], dct['efermi'], dct['s_kvm'])
        if not return_spin:
            out = out[:2]
        return out
    else:
        return dct['eps_nosoc_skn'], dct['efermi_nosoc']


def eigenvalues(calc):
    """Get eigenvalues from gpaw calculator

    Parameters:
    -----------
    calc: obj
        GPAW calculator object

    Returns:
    --------
    e_skn : (ns, nk, nb)-shape array
    """
    rs = range(calc.get_number_of_spins())
    rk = range(len(calc.get_ibz_k_points()))
    e = calc.get_eigenvalues
    return np.asarray([[e(spin=s, kpt=k) for k in rk] for s in rs])


def fermi_level(calc, eps_skn=None, nelectrons=None):
    """
    Parameters:
        calc: GPAW
            GPAW calculator
        eps_skn: ndarray, shape=(ns, nk, nb), optional
            eigenvalues (taken from calc if None)
        nelectrons: float, optional
            number of electrons (taken from calc if None)
    Returns:
        out: float
            fermi level
    """
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()
    if eps_skn is None:
        eps_skn = eigenvalues(calc)
    eps_skn.sort(axis=-1)
    occ = calc.occupations.todict()
    weight_k = calc.get_k_point_weights()

    return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha


# XXX preferably, move this method to anisotropy recipe!
def get_spin_direction(fname='anisotropy_xy.npz'):
    """Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Parameters:
    -----------
    fname : str
        The filename of a datafile containing the xz and yz
        anisotropy energies.

    Returns:
    --------
    theta : float
        Polar angle in radians
    phi : float
        Azimuthal angle in radians
    """

    import os.path as op
    theta = 0
    phi = 0
    if op.isfile(fname):
        data = np.load(fname)
        DE = max(data['dE_zx'], data['dE_zy'])
        if DE > 0:
            theta = np.pi / 2
            if data['dE_zy'] > data['dE_zx']:
                phi = np.pi / 2
    return theta, phi
