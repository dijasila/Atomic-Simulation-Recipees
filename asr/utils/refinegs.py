from gpaw import GPAW
from asr.utils.kpts import get_kpts_size


def nonselfc(kptdens=12, emptybands=20, txt=None):
    """Non self-consistent calculation based on the density in gs.gpw"""
    parstr = '_kptdens«%s»_emptybands«%s»' % (str(kptdens), str(emptybands))

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

    return calc, parstr


def write_refinedgs(calc, outf, parstr):
    if isinstance(outf, str):
        assert outf[-4:] == '.gpw'
    else:
        outf = 'refinedgs' + parstr + '.gpw'
    calc.write(outf)
    return outf


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
    outf : str
        filename of written GPAW calculator object
    """
    if selfc:
        raise NotImplementedError('Someone should implement refinement '
                                  + 'with self-consistency')
    else:
        calc, parstr = nonselfc(*args, **kwargs)
    parstr = '_selfc«%s»%s' % (selfc, parstr)

    if outf:
        outf = write_refinedgs(calc, outf, parstr)

    return calc, outf
