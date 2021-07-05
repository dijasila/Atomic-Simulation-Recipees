"""Deformation potentials."""
from typing import Tuple

import numpy as np
from ase import Atoms

import asr.gs
from asr.calculators import get_calculator_class
from asr.core import (ASRResult, atomsopt, calcopt, command, option,
                      prepare_result)
from asr.magnetic_anisotropy import get_spin_axis
from asr.relax import main as relax
from asr.setup.strains import get_relevant_strains
from asr.setup.strains import main as make_strained_atoms
from asr.utils.gpw2eigs import calc2eigs


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
    vbm: float
    cbm: float
    vbm_nosoc: float
    cbm_nosoc: float
    vacuumlevel: float


# @command('asr.deformationpotentials')
def calculate(atoms: Atoms,
              calculator: dict,
              edge_positions: Tuple[Tuple[int, int, int],
                                    Tuple[int, int, int],
                                    Tuple[int, int],
                                    Tuple[int, int]],
              theta: float,
              phi: float) -> EdgesResult:
    """
    """
    (s1, K1, n1), (s2, K2, n2), (K1soc, n1soc), (K2soc, n2soc) = edge_positions

    calculator = calculator.copy()
    atoms = atoms.copy()

    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)
    atoms.calc = calc
    atoms.get_potential_energy()
    kd = calc.wfs.kd

    k1 = kd.bz2ibz_k[K1]
    k2 = kd.bz2ibz_k[K2]
    if kd.nspins == 1:
        s1 = 0
        s2 = 0
    vbm_nosoc = calc.get_eigenvalues(spin=s1, kpt=k1)[n1]
    cbm_nosoc = calc.get_eigenvalues(spin=s2, kpt=k2)[n2]

    e_km, efermi = calc2eigs(calc,
                             soc=True, theta=theta, phi=phi)
    vbm = e_km[K1soc, n1soc]
    cbm = e_km[K2soc, n2soc]

    assert (atoms.pbc == [1, 1, 0]).all()
    vacuumlevel = calc.get_electrostatic_potential()[:, :, 5].mean()

    return EdgesResult.fromdata(
        vbm=vbm,
        cbm=cbm,
        vbm_nosoc=vbm_nosoc,
        cbm_nosoc=cbm_nosoc,
        vacuumlevel=vacuumlevel)


@prepare_result
class Result(ASRResult):
    edges: np.ndarray
    deformation_potentials: np.ndarray
    edges_nosoc: np.ndarray
    deformation_potentials_nosoc: np.ndarray

    key_descriptions = dict(
        edges='Array of band edges',
        deformation_potentials='Deformation potentials',
        edges_nosoc='Array of band edges without SOC',
        deformation_potentials_nosoc='Deformation potentials without SOC')

    formats = {'webpanel2': webpanel}


@command('asr.deformationpotentials')
@atomsopt
@calcopt
@option('--strain-percent', help='Strain fraction.', type=float)
def main(
        atoms: Atoms,
        calculator: dict = asr.gs.main.defaults.calculator,  # mutable ?????
        strain_percent: float = 1.0) -> Result:
    """Calculate deformation potentials.

    Calculate the deformation potential both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary.
    """
    from asr.gs import calculate as gscalculate
    from asr.gs import main as groundstate
    strains = [-strain_percent, strain_percent]

    ij = get_relevant_strains(atoms.pbc)

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    # Edges have dimension (2, 6, 2) =
    # (#strains_percentages, #strains, (vbm, cbm))
    # Because np.polyfit likes that
    edges_pin = np.zeros((2, 6, 2), float)
    edges_nosoc_pin = np.zeros((2, 6, 2), float)

    gsresults = groundstate(
        atoms=atoms,
        calculator=calculator)

    gscalc = gscalculate(
        atoms=atoms,
        calculator=calculator)

    calc = gscalc.calculation.load()
    kd = calc.wfs.kd

    _, K1soc, n1soc = gsresults['skn1']
    _, K2soc, n2soc = gsresults['skn2']

    s1, k1, n1 = gsresults['gaps_nosoc']['skn1']
    s2, k2, n2 = gsresults['gaps_nosoc']['skn2']

    # Convert from IBZ to full BZ index:
    K1 = kd.ibz2bz_k[k1]
    K2 = kd.ibz2bz_k[k2]

    theta, phi = get_spin_axis(atoms, calculator=calculator)

    calculator = calculator.copy()
    calculator['kpts'] = {'size': kd.N_c, 'gamma': True}

    edge_positions = [(s1, K1, n1),
                      (s2, K2, n2),
                      (K1soc, n1soc),
                      (K2soc, n2soc)]

    for i, j in ij:
        for ip, strain in enumerate(strains):
            strained_atoms = make_strained_atoms(
                atoms,
                strain_percent=strain,
                i=i, j=j)

            relaxresults = relax(
                atoms=strained_atoms,
                calculator=calculator,
                fixcell=True)

            edges = calculate(relaxresults.atoms,
                              calculator,
                              edge_positions,
                              theta, phi)

            evac = edges.vacuumlevel

            v = ij_to_voigt[i][j]
            edges_pin[ip, v, 0] = edges.vbm - evac
            edges_nosoc_pin[ip, v, 0] = edges.vbm_nosoc - evac
            edges_pin[ip, v, 1] = edges.cbm - evac
            edges_nosoc_pin[ip, v, 1] = edges.cbm_nosoc - evac

    results = {'edges': edges_pin,
               'edges_nosoc': edges_nosoc_pin}

    for soc in (True, False):
        if soc:
            edges_pin = edges_pin
        else:
            edges_pin = edges_nosoc_pin

        deformation_potentials = np.zeros(np.shape(edges_pin)[1:])
        for idx, band_edge in enumerate(['vbm', 'cbm']):
            D = np.polyfit(strains, edges_pin[:, :, idx], 1)[0] * 100
            deformation_potentials[:, idx] = D
        results[['deformation_potentials_nosoc',
                 'deformation_potentials'][soc]] = \
            deformation_potentials.tolist()

    return results


if __name__ == '__main__':
    main.cli()
