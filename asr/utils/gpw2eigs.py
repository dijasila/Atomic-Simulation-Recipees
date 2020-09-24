

def calc2eigs(calc, ranks, soc=True,
              return_spin=False,
              theta=0, phi=0, symmetry_tolerance=1e-7,
              width=None):
    from gpaw.spinorbit import get_spinorbit_eigenvalues
    from ase.parallel import broadcast
    import numpy as np
    from .symmetry import restrict_spin_projection_2d
    from .calculator_utils import get_eigenvalues, fermi_level
    from .symmetry import _atoms2symmetry_gpaw

    dct = None
    if calc.world.rank in ranks:
        bands = range(calc.get_number_of_bands())
        eps_nosoc_skn = get_eigenvalues(calc)[..., bands]
        efermi_nosoc = calc.get_fermi_level()
        dct = {'eps_nosoc_skn': eps_nosoc_skn,
               'efermi_nosoc': efermi_nosoc}
        if soc:
            eps_mk, s_kvm = get_spinorbit_eigenvalues(calc, bands=bands,
                                                      theta=theta,
                                                      phi=phi,
                                                      return_spin=True)
            eps_km = eps_mk.T
            eps_km.sort(axis=-1)
            efermi = fermi_level(calc, eps_km[np.newaxis], nspins=2)
            symmetry = _atoms2symmetry_gpaw(calc.atoms,
                                            tolerance=symmetry_tolerance)
            ibzk_kc = calc.get_ibz_k_points()
            # For magnetic systems we know some more about the spin
            # projections
            if calc.get_number_of_spins() == 1:
                # Inversion + time reversal symmetry forces degenerates spins
                if symmetry.has_inversion:
                    s_kvm[:] = 0.0
                else:
                    # For 2D we try to find materials where
                    # spins are restricted to inplane spins
                    if np.sum(calc.atoms.pbc).astype(int) == 2:
                        for ik, kpt in enumerate(ibzk_kc):
                            s_vm = restrict_spin_projection_2d(kpt,
                                                               symmetry.op_scc,
                                                               s_kvm[ik])
                            s_kvm[ik] = s_vm
            dct.update({'eps_km': eps_km,
                        'efermi': efermi,
                        's_kvm': s_kvm})

    dct = broadcast(dct, root=0, comm=calc.world)
    if soc is None:
        raise NotImplementedError('soc=None is not implemented')

    if soc:
        out = (dct['eps_km'], dct['efermi'], dct['s_kvm'])
        if not return_spin:
            out = out[:2]
        return out
    else:
        if not return_spin:
            return dct['eps_nosoc_skn'], dct['efermi_nosoc']
        return dct['eps_nosoc_skn'], dct['efermi_nosoc'], dct['s_kvm']


def gpw2eigs(gpw, soc=True, return_spin=False,
             theta=0, phi=0, symmetry_tolerance=1e-7):
    """Give the eigenvalues w or w/o spinorbit coupling and the corresponding fermi energy.

    Parameters
    ----------
        gpw (str): gpw filename
        soc : None, bool
            use spinorbit coupling if None it returns both w and w/o
        bands : slice, list of ints or None
            None gives parameters.convergence.bands if possible else all bands

    Returns
    -------
    dict or e_skn, efermi
        containg eigenvalues and fermi levels w and w/o spinorbit coupling
    """
    from gpaw import GPAW
    from gpaw import mpi
    ranks = [0]
    calc = GPAW(gpw, txt=None, communicator=mpi.serial_comm)
    return calc2eigs(calc, soc=soc, return_spin=return_spin,
                     theta=theta, phi=phi,
                     ranks=ranks,
                     symmetry_tolerance=symmetry_tolerance)
