from pathlib import Path

from asr.utils.kpts import get_kpts_size


def nonselfc(txt=None, kptdensity=20.0, emptybands=20):
    """Non self-consistent calculation based on the density in gs.gpw."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw', txt=None)

    kpts = get_kpts_size(atoms=calc.atoms, kptdensity=kptdensity)
    convbands = int(emptybands / 2)
    calc = calc.fixed_density(nbands=-emptybands,
                              txt=txt,
                              kpts=kpts,
                              convergence={'bands': -convbands})

    return calc


def get_filenames(gpw, txt, selfc=False, **kwargs):
    """Get file names as specified by input."""
    parstr = get_parstr(selfc=selfc, **kwargs)

    if isinstance(gpw, str):
        if gpw == 'default':
            gpw = f'refinedgs_{parstr}.gpw'
        assert gpw[-4:] == '.gpw'
    else:
        assert gpw is None

    if isinstance(txt, str):
        if txt == 'default':
            txt = f'refinedgs_{parstr}.txt'
    else:
        assert txt is None

    return gpw, txt


def get_parstr(selfc=False, **kwargs):
    """Get parameter string, specifying how the ground state is refined."""
    parstr = 'selfc«%s»' % str(selfc)

    for kw in ['kptdensity', 'emptybands']:
        parstr += '_%s«%s»' % (kw, str(kwargs[kw]))

    return parstr


def refinegs(selfc=False, gpw=None, txt=None, **kwargs):
    """Refine the ground state calculation.

    Parameters
    ----------
    selfc : bool
        Perform new self-consistency cycle to refine also the density
    gpw : str
        Write the refined ground state as a .gpw file.
        If 'default' is specified, use f'refinedgs_{parstr}.gpw' as file name.
        If another string is specified, use that as file name.
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
    from gpaw import GPAW
    gpw, txt = get_filenames(gpw, txt, selfc=selfc, **kwargs)
    if gpw and Path(gpw).is_file():
        calc = GPAW(gpw, txt=None)
    else:
        if selfc:
            raise NotImplementedError('Someone should implement refinement '
                                      + 'with self-consistency')
        else:
            calc = nonselfc(txt=txt, **kwargs)

        if gpw:
            calc.write(gpw)

    return calc, gpw
