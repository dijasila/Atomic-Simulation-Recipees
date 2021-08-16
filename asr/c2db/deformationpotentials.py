"""Deformation potentials."""
# from typing import Tuple
from functools import partial

import numpy as np
from ase import Atoms

import asr.c2db.gs
from asr.calculators import get_calculator_class
from asr.core import (ASRResult, atomsopt, calcopt, command, option,
                      prepare_result)
from asr.c2db.magnetic_anisotropy import get_spin_axis
from asr.c2db.relax import main as relax
from asr.setup.strains import get_relevant_strains
from asr.setup.strains import main as make_strained_atoms
from asr.utils.gpw2eigs import calc2eigs
from asr.c2db.gs import main as groundstate, calculate as gscalculate


def webpanel(result, context):
    data = context.result

    defpot = data['deformation_potentials']
    vbmdef = (defpot[0, 0] + defpot[1, 0]) / 2
    cbmdef = (defpot[0, 1] + defpot[1, 1]) / 2
    rows = [['Deformation potential at VBM', f'{vbmdef:0.2f} eV'],
            ['Deformation potential at CBM', f'{cbmdef:0.2f} eV']]

    panel = {'title': f'Basic electronic properties ({context.xcname})',
             'columns': [[{'type': 'table',
                           'header': ['Property', ''],
                           'rows': rows}]],
             'sort': 11}
    return [panel]


@prepare_result
class EdgesResult(ASRResult):
    evbm: float
    ecbm: float
    vacuumlevel: float


@command('asr.deformationpotentials@calculate')
def calculate(atoms: Atoms,
              calculator: dict,
              vbm_position: dict,
              cbm_position: dict,
              angles: dict) -> EdgesResult:
    calculator = calculator.copy()
    atoms = atoms.copy()

    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)
    atoms.calc = calc
    atoms.get_potential_energy()

    e_km, efermi = calc2eigs(calc, soc=True, **angles)
    evbm = e_km[vbm_position['k'], vbm_position['n']]
    ecbm = e_km[cbm_position['k'], cbm_position['n']]

    assert (atoms.pbc == [1, 1, 0]).all()
    vacuumlevel = calc.get_electrostatic_potential()[:, :, 5].mean()

    return EdgesResult.fromdata(
        evbm=evbm,
        ecbm=ecbm,
        vacuumlevel=vacuumlevel)


@prepare_result
class Result(ASRResult):
    edges: np.ndarray
    deformation_potentials: np.ndarray

    key_descriptions = dict(
        edges='Array of band edges',
        deformation_potentials='Deformation potentials')

    formats = {'webpanel': webpanel}


def _main(atoms: Atoms,
          vbm_position,
          cbm_position,
          relax_atoms,
          calculate_band_edges,
          strains=[-1.0, 1.0]):
    """Calculate deformation potentials.

    Calculate the deformation potential both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary.
    """
    ij = get_relevant_strains(atoms.pbc)

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    # Edges have dimension (2, 6, 2) =
    # (#strains_percentages, #strains, (vbm, cbm))
    # Because np.polyfit likes that
    edges_pin = np.zeros((2, 6, 2), float)

    for i, j in ij:
        for ip, strain in enumerate(strains):
            strained_atoms = make_strained_atoms(
                atoms,
                strain_percent=strain,
                i=i, j=j)

            relaxresults = relax_atoms(strained_atoms)

            edges = calculate_band_edges(
                relaxresults.atoms,
                vbm_position,
                cbm_position)

            evac = edges['vacuumlevel']

            v = ij_to_voigt[i][j]
            edges_pin[ip, v, 0] = edges['evbm'] - evac
            edges_pin[ip, v, 1] = edges['ecbm'] - evac

    deformation_potentials = np.zeros(np.shape(edges_pin)[1:])
    for idx, band_edge in enumerate(['vbm', 'cbm']):
        D = np.polyfit(strains, edges_pin[:, :, idx], 1)[0] * 100
        deformation_potentials[:, idx] = D

    return edges_pin, deformation_potentials


def main(atoms: Atoms,
         calculator: dict = asr.c2db.gs.main.defaults.calculator,
         strain_percent: float = 1.0) -> Result:
    """Calculate deformation potentials.

    Calculate the deformation potential both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary.
    """
    strains = [-strain_percent, strain_percent]

    gsresults = groundstate(
        atoms=atoms,
        calculator=calculator)

    gscalc = gscalculate(
        atoms=atoms,
        calculator=calculator)

    calc = gscalc.calculation.load()
    size = calc.wfs.kd.N_c

    # K1 and K2 are indices of the unreduced BZ
    _, K1, n1 = gsresults['skn1']
    _, K2, n2 = gsresults['skn2']
    vbm_position = dict(K=K1, n=n1)
    cbm_position = dict(K=K2, n=n2)

    theta, phi = get_spin_axis(atoms, calculator=calculator)

    calculator = calculator.copy()
    calculator['kpts'] = {'size': size, 'gamma': True}

    edges_pin, deformation_potentials = _main(
        atoms,
        vbm_position,
        cbm_position,
        partial(relax, calculator=calculator, fixcell=True),
        partial(calculate,
                calculator=calculator,
                theta=theta, phi=phi),
        strains)

    return Result.fromdata(
        edges=edges_pin,
        deformation_potentials=deformation_potentials)


if __name__ == '__main__':
    main.cli()
