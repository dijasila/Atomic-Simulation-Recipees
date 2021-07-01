"""Deformation potentials."""
from typing import Tuple

import numpy as np
from ase import Atoms

from asr.core import ASRResult, atomsopt, calcopt, command, prepare_result
from asr.gs import calculate as gscalculate
from asr.gs import main as groundstate
from asr.relax import main as relax
from asr.setup.strains import get_relevant_strains
from asr.setup.strains import main as make_strained_atoms


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
class Result(ASRResult):
    formats = {'webpanel2': webpanel}


def get_edges(atoms,
              calculator,
              edge_positions):
    """
    sKn1: Tuple[int, int, int],
    sKn2: Tuple[int, int, int]) -> Tuple[float, float]
    """
    e1, e2, e1soc, e2soc = ...

    s1, K1, n1 = sKn1
    s2, K2, n2 = sKn2
    k1 = kd.bz2ibz_k[K1]
    k2 = kd.bz2ibz_k[K2]
    if kd.nspins == 1:
        s1 = 0
        s2 = 0
    e1 = calc.get_eigenvalues(spin=s1, kpt=k1)[n1]
    e2 = calc.get_eigenvalues(spin=s2, kpt=k2)[n2]
    return e1, e2


@command('asr.deformationpotentials')
@atomsopt
@calcopt
def main(
        atoms: Atoms,
        calculator: dict = groundstate.defaults.calculator) -> Result:
    """Calculate deformation potentials.

    Calculate the deformation potential both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary.
    """
    strains = [-1.0, 1.0]

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

    s1, k1, n1 = gsresults['gaps_nosoc']['skn1']
    s2, k2, n2 = gsresults['gaps_nosoc']['skn2']
    # Convert from IBZ to full BZ index:
    K1 = kd.ibz2bz_k[k1]
    K2 = kd.ibz2bz_k[k2]

    calculator['kpts'] = {'size': kd.N_c, 'gamma': True}

    edge_positions = [(s1, K1, n1),
                      (s2, K2, n2),
                      (spin-projection?, K1soc, n1soc),
                      ...]

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

            e1, e2, e1soc, e2soc = get_edges(relaxresults.atoms,
                                             calculator,
                                             edge_positions)

            evac = gsresults['evac']
            edges_pin[ip, ij_to_voigt[i][j], 0] = e1soc - evac
            edges_nosoc_pin[ip, ij_to_voigt[i][j], 0] = e1 - evac
            edges_pin[ip, ij_to_voigt[i][j], 1] = e2soc - evac
            edges_nosoc_pin[ip, ij_to_voigt[i][j], 1] = e2 - evac

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
