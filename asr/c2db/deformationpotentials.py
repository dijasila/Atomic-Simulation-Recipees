"""Deformation potentials."""
from typing import List
import numpy as np

from ase import Atoms

from asr.core import (
    command, option, ASRResult, prepare_result, atomsopt, calcopt
)

from asr.setup.strains import main as make_strained_atoms
from asr.setup.strains import get_relevant_strains
from asr.c2db.gs import main as groundstate


def webpanel(result, context):
    data = context.result

    defpot = data['deformation_potentials']
    vbmdef = (defpot[0, 0] + defpot[1, 0]) / 2
    cbmdef = (defpot[0, 1] + defpot[1, 1]) / 2
    rows = [['Uniaxial deformation potential at VBM', f'{vbmdef:0.2f} eV'],
            ['Uniaxial deformation potential at CBM', f'{cbmdef:0.2f} eV']]
    panel = {'title': f'Basic electronic properties ({context.xcname})',
             'columns': [[{'type': 'table',
                           'header': ['Property', ''],
                           'rows': rows}]],
             'sort': 11}
    return [panel]


@prepare_result
class Result(ASRResult):
    formats = {'webpanel2': webpanel}


@command('asr.c2db.deformationpotentials')
@atomsopt
@calcopt
@option('--strains', help='Strain percentages', type=float)
@option('--ktol',
        help='Distance in k-space that extremum is allowed to move.',
        type=float)
def main(
        atoms: Atoms,
        calculator: dict = groundstate.defaults.calculator,
        strains: List[float] = [-1.0, 0.0, 1.0], ktol: float = 0.1) -> Result:
    """Calculate deformation potentials.

    Calculate the deformation potential both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary.
    """
    strains = sorted(strains)
    ij = get_relevant_strains(atoms.pbc)

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    # Edges have dimension (3, 6, 2) =
    # (#strains_percentages, #strains, (vbm, cbm))
    # Because np.polyfit likes that
    edges_pin = np.zeros((3, 6, 2), float)
    edges_nosoc_pin = np.zeros((3, 6, 2), float)

    gsresults = groundstate(
        atoms=atoms,
        calculator=calculator,
    )

    k0_vbm_c = gsresults['k_vbm_c']
    k0_cbm_c = gsresults['k_cbm_c']

    for i, j in ij:
        for ip, strain in enumerate(strains):
            strained_atoms = make_strained_atoms(
                atoms,
                strain_percent=strain,
                i=i, j=j)

            gsresults = groundstate(
                atoms=strained_atoms,
                calculator=calculator,
            )
            k_vbm_c = gsresults['k_vbm_c']
            k_cbm_c = gsresults['k_cbm_c']
            difference = k_vbm_c - k0_vbm_c
            difference -= np.round(difference)
            assert (np.abs(difference) < ktol).all(), \
                (f'i={i} j={j} strain={strain}: VBM has '
                 f'changed location in reciprocal space upon straining. '
                 f'{k0_vbm_c} -> {k_vbm_c} (Delta_c={difference})')
            difference = k_cbm_c - k0_cbm_c
            difference -= np.round(difference)
            assert (np.abs(difference) < ktol).all(), \
                (f'i={i} j={j} strain={strain}: CBM has '
                 f'changed location in reciprocal space upon straining. '
                 f'{k0_cbm_c} -> {k_cbm_c} (Delta_c={difference})')
            evac = gsresults['evac']
            edges_pin[ip, ij_to_voigt[i][j], 0] = gsresults['vbm'] - evac
            edges_nosoc_pin[ip, ij_to_voigt[i][j], 0] = \
                gsresults['gaps_nosoc']['vbm'] - evac
            edges_pin[ip, ij_to_voigt[i][j], 1] = gsresults['cbm'] - evac
            edges_nosoc_pin[ip, ij_to_voigt[i][j], 1] = \
                gsresults['gaps_nosoc']['cbm'] - evac

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
