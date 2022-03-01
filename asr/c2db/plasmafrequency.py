"""Plasma frequency."""
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.parallel import world
from ase.units import Hartree, Bohr

from asr.core import (
    command, option, ASRResult, prepare_result, atomsopt, calcopt)
import typing
from asr.utils.kpts import get_kpts_size
from asr.c2db.gs import calculate as gscalculate, main as gsmain


# XXX The plasmafrequency recipe should not be two steps. We don't
# want to keep the large gpw since it can potentially be really large.
# Therefore I have degraded the calculate step to a simple function.

def calculate(
        atoms: Atoms,
        calculator: dict = gscalculate.defaults.calculator,
        kptdensity: float = 20,
) -> ASRResult:
    """Calculate excited states for polarizability calculation."""
    res = gscalculate(atoms=atoms, calculator=calculator)
    # We want the gap for the webpanel, so explicitly call gs main:
    gsmain(atoms=atoms, calculator=calculator)
    calc_old = res.calculation.load()
    kpts = get_kpts_size(atoms=calc_old.atoms, kptdensity=kptdensity)
    nval = calc_old.wfs.nvalence
    filename = "es_plasma.gpw"
    try:
        calc = res.calculation.load(
            fixdensity=True,
            kpts=kpts,
            nbands=2 * nval,
            txt='gsplasma.txt',
        )
        calc.get_potential_energy()
        calc.write(filename, 'all')
    except Exception:
        if world.rank == 0:
            es_file = Path(filename)
            if es_file.is_file():
                es_file.unlink()
        world.barrier()

    return filename


def webpanel(result, context):
    from asr.database.browser import table

    gsresults = context.gs_results()
    if gsresults['gap'] > 0.01:
        return []

    assert 'plasmafrequency_x' in result
    plasmatable = table(result, 'Property', [
        'plasmafrequency_x', 'plasmafrequency_y'], context.descriptions)

    panel = {'title': 'Optical polarizability (RPA)',
             'columns': [[], [plasmatable]]}
    return [panel]


@prepare_result
class Result(ASRResult):

    plasmafreq_vv: typing.List[typing.List[float]]
    plasmafrequency_x: float
    plasmafrequency_y: float

    key_descriptions = {
        "plasmafreq_vv": "Plasma frequency tensor [Hartree]",
        "plasmafrequency_x": "KVP: 2D plasma frequency (x)"
        "[`eV/Å^0.5`]",
        "plasmafrequency_y": "KVP: 2D plasma frequency (y)"
        "[`eV/Å^0.5`]",
    }
    formats = {'webpanel2': webpanel}


@command('asr.c2db.plasmafrequency')
@atomsopt
@calcopt
@option('--kptdensity', help='k-point density', type=float)
@option('--tetra', is_flag=True,
        help='Use tetrahedron integration')
def main(
        atoms: Atoms,
        calculator: dict = gscalculate.defaults.calculator,
        kptdensity: float = 20,
        tetra: bool = True,
) -> Result:
    """Calculate polarizability."""
    from gpaw.response.df import DielectricFunction

    gpwfile = calculate(
        atoms=atoms,
        calculator=calculator,
        kptdensity=kptdensity,
    )
    nd = sum(atoms.pbc)
    if not nd == 2:
        raise AssertionError('Plasmafrequency recipe only implemented for 2D')

    if tetra:
        kwargs = {'truncation': '2D',
                  'eta': 0.05,
                  'domega0': 0.2,
                  'integrationmode': 'tetrahedron integration',
                  'ecut': 1,
                  'pbc': [True, True, False]}
    else:
        kwargs = {'truncation': '2D',
                  'eta': 0.05,
                  'domega0': 0.2,
                  'ecut': 1}

    try:
        df = DielectricFunction(gpwfile, **kwargs)
        df.get_polarizability(q_c=[0, 0, 0], direction='x',
                              pbc=[True, True, False],
                              filename=None)
    finally:
        world.barrier()
        if world.rank == 0:
            es_file = Path(gpwfile)
            es_file.unlink()
    plasmafreq_vv = df.chi0.plasmafreq_vv.real
    data = {'plasmafreq_vv': plasmafreq_vv}

    if nd == 2:
        wp2_v = np.linalg.eigvalsh(plasmafreq_vv[:2, :2])
        L = atoms.cell[2, 2] / Bohr
        plasmafreq_v = (np.sqrt(wp2_v * L / 2) * Hartree * Bohr**0.5)
        data['plasmafrequency_x'] = plasmafreq_v[0].real
        data['plasmafrequency_y'] = plasmafreq_v[1].real

    return data
