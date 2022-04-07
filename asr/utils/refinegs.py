from asr.utils.kpts import get_kpts_size


def refinegs(atoms, calculator, txt=None, kptdensity=20.0, emptybands=20):
    """Refine the ground state calculation.

    Returns GPAW calculator with fixed density.
    """

    from asr.c2db.gs import calculate
    res = calculate(atoms=atoms, calculator=calculator)
    calc = res.calculation.load()

    kpts = get_kpts_size(atoms=calc.atoms, kptdensity=kptdensity)
    convbands = int(emptybands / 2)
    calc = calc.fixed_density(nbands=-emptybands,
                              txt=txt,
                              kpts=kpts,
                              convergence={'bands': -convbands})

    return calc
