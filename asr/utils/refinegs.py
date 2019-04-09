from pathlib import Path

from gpaw import GPAW

from asr.utils.kpts import get_kpts_size


def nonselfc(kptdens=12, emptybands=20, txt=None):
    """Non self-consistent calculation based on the density in gs.gpw"""
    calc = GPAW('gs.gpw', txt=None)
    spinpol = calc.get_spin_polarized()

    kpts = get_kpts_size(atoms=calc.atoms, density=kptdens)
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


def write_refinedgs(calc, outf, parstr):
    if isinstance(outf, str):
        assert outf[-4:] == '.gpw'
    else:
        outf = 'refinedgs' + parstr + '.gpw'
    calc.write(outf)
    return outf


def get_gpw(selfc, outf, **kwargs):
    """Get filename corresponding to the wanted refinement"""
    if isinstance(outf, str):
        assert outf[-4:] == '.gpw'
        gpw = outf
    else:
        parstr = '_selfc«%s»' % str(selfc)
        for kw in ['kptdens', 'emptybands']:
            parstr += '_%s«%s»' % (kw, str(kwargs[kw]))
        gpw = 'refinedgs' + parstr + '.gpw'
    return gpw


def refinegs(selfc=False, outf=False, *args, **kwargs):
    """Refine the ground state calculation

    Parameters:
    -----------
    selfc : bool
        Perform new self-consistency cycle to refine also the density
    outf : bool, str
        Write the refined ground state as a GPAW calculator object.
        If a string is specified, use that as file name, otherwise use the
        ('refinedgs%s.gpw' % parstr) convention.

    Returns:
    --------
    calc : obj
        GPAW calculator object
    gpw : str
        filename of written GPAW calculator object
    """
    gpw = get_gpw(selfc, outf, **kwargs)
    if Path(gpw).is_file():
        calc = GPAW(gpw, txt=None)
    else:
        if selfc:
            raise NotImplementedError('Someone should implement refinement '
                                      + 'with self-consistency')
        else:
            calc = nonselfc(*args, **kwargs)

        if outf:
            calc.write(gpw)

    return calc, gpw
