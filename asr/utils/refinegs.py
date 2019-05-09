from gpaw import GPAW
from asr.utils.kpts import get_kpts_size


def nonsc(kdens=12, emptybands=20, txt=None):
    """Non self-consistent calculation with dense k-point sampling
       based on the density in gs.gpw
    """

    calc = GPAW('gs.gpw', txt=None)
    spinpol = calc.get_spin_polarized()

    kpts = get_kpts_size(atoms=calc.atoms, density=kdens)
    convbands = int(emptybands / 2)
    calc.set(nbands=-emptybands,
             txt=txt,
             fixdensity=True,
             kpts=kpts,
             convergence={'bands': -convbands})

    if spinpol:
        calc.set(symmetry='off')  # due to soc

    calc.get_potential_energy()
    return calc


def refinegs(sc=False, *args, **kwargs):
    """Refine the ground state calculation
    
    Parameters:
    -----------
    sc : bool
        Do not refine the density, but do a one-shot calculation instead
    
    Returns:
    --------
    calc : obj
        GPAW calculator object
    """
    if sc:
        raise NotImplementedError('Someone should implement refinement '
                                  + 'with self-consistency')
    else:
        return nonsc(*args, **kwargs)
