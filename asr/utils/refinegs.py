from pathlib import Path

from asr.utils.kpts import get_kpts_size


def nonselfc(atoms, calculator, txt=None, kptdensity=20.0, emptybands=20):
    """Non self-consistent calculation based on the density in gs.gpw."""
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


def get_parstr(selfc=False, **kwargs):
    """Get parameter string, specifying how the ground state is refined."""
    parstr = 'selfc«%s»' % str(selfc)

    for kw in ['kptdensity', 'emptybands']:
        parstr += '_%s«%s»' % (kw, str(kwargs[kw]))

    return parstr


def refinegs(atoms, calculator, txt, **kwargs):
    """Refine the ground state calculation.

    Parameters
    ----------
    txt : str
        Write the GPAW output to a .txt file.
        If 'default' is specified, use f'refinedgs_{parstr}.txt' as file name.
        If another string is specified, use that as file name.

    Returns
    -------
    calc : obj
        GPAW calculator object
    gpw : str
        filename of written GPAW calculator object
    """
    calc = nonselfc(atoms, calculator, txt=txt, **kwargs)
    return calc
