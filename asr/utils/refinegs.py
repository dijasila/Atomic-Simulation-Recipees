from asr.utils.kpts import get_kpts_size


def refinegs(gsresult, txt=None, kptdensity=20.0, emptybands=20):
    """Refine the ground state calculation.

    Returns GPAW calculator with fixed density.
    """
    calc = gsresult.calculation.load()

    kpts = get_kpts_size(atoms=calc.atoms, kptdensity=kptdensity)
    convbands = int(emptybands / 2)
    calc = calc.fixed_density(nbands=-emptybands,
                              txt=txt,
                              kpts=kpts,
                              convergence={'bands': -convbands})

    return calc
