import os.path as op
from gpaw import GPAW
from c2db.utils import get_kpts_size


def nonsc(kdens=12, emptybands=20, outname='densk'):
    """Non self-consistent calculation with dense k-point sampling
       based on the density in gs.gpw
    """
    if op.isfile(outname + '.gpw'):
        return GPAW(outname + '.gpw', txt=None)

    calc = GPAW('gs.gpw', txt=None)
    spinpol = calc.get_spin_polarized()

    kpts = get_kpts_size(atoms=calc.atoms, density=kdens)
    convbands = int(emptybands / 2)
    calc.set(nbands=-emptybands,
             txt=outname + '.txt',
             fixdensity=True,
             kpts=kpts,
             convergence={'bands': -convbands})

    if spinpol:
        calc.set(symmetry='off')  # due to soc

    calc.get_potential_energy()
    calc.write(outname + '.gpw')
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
        return nonsc(*args, **kwargs)
    else:
        raise NotImplementedError('Someone should implement refinement '
                                  + 'with self-consistency')
