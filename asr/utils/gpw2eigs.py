def get_spin_direction(fname="anisotropy_xy.npz"):
    """Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Parameters:
        fname: The filename of a datafile containing the xz and yz
            anisotropy energies.

    Returns:
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
    """
    import os
    import numpy as np
    theta = 0
    phi = 0
    if os.path.isfile(fname):
        data = np.load(fname)
        DE = max(data["dE_zx"], data["dE_zy"])
        if DE > 0:
            theta = np.pi / 2
            if data["dE_zy"] > data["dE_zx"]:
                phi = np.pi / 2
    return theta, phi


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
    """Get Fermi level from calculation.

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
    weight_k = calc.get_k_point_weights()
    return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha


def calc2eigs(calc, ranks, soc=True, bands=None, return_spin=False,
              optimal_spin_direction=False):
    from gpaw.spinorbit import get_spinorbit_eigenvalues
    from gpaw import mpi
    from ase.parallel import broadcast
    import numpy as np
    dct = None
    if mpi.world.rank in ranks:
        theta = 0
        phi = 0
        if optimal_spin_direction:
            theta, phi = get_spin_direction()
        if bands is None:
            n2 = calc.todict().get("convergence", {}).get("bands")
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
                             nelectrons=2 * calc.get_number_of_electrons())
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


def gpw2eigs(gpw, soc=True, bands=None, return_spin=False,
             optimal_spin_direction=False):
    """Give the eigenvalues w or w/o spinorbit coupling and the corresponding
    fermi energy

    Parameters:
        gpw (str): gpw filename
        soc : None, bool
            use spinorbit coupling if None it returns both w and w/o
        optimal_spin_direction : bool
            If True, use get_spin_direction to calculate the spin direction
            for the SOC
        bands : slice, list of ints or None
            None gives parameters.convergence.bands if possible else all bands

    Returns : dict or e_skn, efermi
        containg eigenvalues and fermi levels w and w/o spinorbit coupling
    """
    from gpaw import GPAW
    from gpaw import mpi
    ranks = [0]
    calc = GPAW(gpw, txt=None, communicator=mpi.serial_comm)
    return calc2eigs(calc, soc=True, bands=None, return_spin=False,
                     optimal_spin_direction=False, ranks=ranks)
