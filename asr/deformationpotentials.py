"""Deformation potentials."""
import numpy as np
from collections import OrderedDict

from ase.io.jsonio import read_json

from asr.core import command, option
from asr.utils.gpw2eigs import calc2eigs
from asr.result.resultdata import DefPotsResult, get_table_row


def get_special_kpts(atoms):
    """Obtain the high-symmetry k-points.

    If the band edges of the unstrained material are found away
    from any of the special points, the corresponding
    k-points will be added to the list

    ISSUE:
    There is no way to obtain the special k-points from atoms.cell.reciprocal(),
    which is the correct reciprocal cell (respects vector orthogonality
    wrt. direct cell) and the one used by GPAW when mapping the IBZ.

    atoms.cell.bandpath can return the special points in fractional coordinates
    of bandpath.icell, which is NOT the conventional reciprocal cell.
    However, when scaled by bandpath.icell, the absolute coordinates will
    correspond to the actual special points.

    If these fractional coordinates are used in a calculation,
    GPAW will use cell.reciprocal() and NOT bandpath.icell,
    hence the fractional coordinates will not necessarily
    map to special points.

    Here we obtain the absolute coordinates from bandpath and then convert
    them to fractional coordinates of cell.reciprocal, to ensure that the
    k-points passed to GPAW will actually correspond to special points.
    """
    icell = atoms.cell.reciprocal()
    bp = atoms.cell.bandpath(pbc=atoms.pbc, npoints=0)
    spec_kpts = bp.special_points

    labels = []
    coords = []

    for lab, kpt in spec_kpts.items():
        spec_bp = np.asarray(kpt)
        spec_abs = np.dot(spec_bp, bp.icell)
        kpt_rescaled = np.dot(spec_abs, np.linalg.inv(icell))
        labels.append(lab)
        coords.append(kpt_rescaled)

    return labels, coords


def get_relevant_kpts(calc, folder, soc):
    """Include VBM and CBM into the k-point list."""
    from ase.dft.bandgap import bandgap

    eigs, efermi = calc2eigs(calc, soc=soc)

    if soc:
        kpts = calc.get_bz_k_points()
    else:
        eigs = eigs[0]
        kpts = calc.get_ibz_k_points()

    gap, vbm, cbm = bandgap(
        eigenvalues=eigs,
        efermi=efermi,
        output=None
    )

    if gap == 0.0:
        raise ValueError("""\
            Deformation potentials cannot be defined for metals!
            Terminating recipe...""")

    evac = evac_from_results(folder)
    evbm = eigs[vbm[0]][vbm[1]] - evac
    ecbm = eigs[cbm[0]][cbm[1]] - evac

    labels, coords = get_special_kpts(calc.atoms)
    kpoints = {label: coord for label, coord in zip(labels, coords)}
    edges = gpaw_get_edges('.', coords, soc)

    for i, e in enumerate(edges):
        if e[0] > evbm or abs(e[0] - evbm) < 0.01:
            kpoints['VBM'] = coords[i]
        if e[1] < ecbm or abs(e[1] - ecbm) < 0.01:
            kpoints['CBM'] = coords[i]

    if 'VBM' not in kpoints.keys():
        kpoints['VBM'] = kpts[vbm[0]]
    if 'CBM' not in kpoints.keys():
        kpoints['CBM'] = kpts[cbm[0]]

    return kpoints


def evac_from_results(folder):
    result = read_json(f'{folder}/results-asr.gs.json')
    return result['kwargs']['data']['evac']


def gpaw_get_edges(folder, kpts, soc):
    """Obtain the edge states at the different k-points.

    Returns, for each k-point included in the calculation,
    the top eigenvalue of the valence band and
    the bottom eigenvalue of the conduction band."""

    from gpaw import GPAW

    gpw = GPAW(f'{folder}/gs.gpw').fixed_density(
        kpts=kpts,
        symmetry='off',
        txt=None
    )
    gpw.get_potential_energy()
    all_eigs, efermi = calc2eigs(gpw, soc=soc)
    evac = evac_from_results(folder)

    # This will take care of the spin polarization
    if not soc:
        all_eigs = np.hstack(all_eigs)

    edges = np.zeros((len(all_eigs), 2))
    for i, eigs_k in enumerate(all_eigs):
        vb = [eig for eig in eigs_k if eig - efermi < 0]
        cb = [eig for eig in eigs_k if eig - efermi > 0]
        edges[i, 0] = max(vb)
        edges[i, 1] = min(cb)
    return edges - evac


def _main(pbc, kpts, get_edges, strain, soc):
    from asr.setup.strains import (get_relevant_strains,
                                   get_strained_folder_name)
    ijlabels = {
        (0, 0): 'xx',
        (1, 1): 'yy',
        (2, 2): 'zz',
        (0, 1): 'xy',
        (0, 2): 'xz',
        (1, 2): 'yz',
    }
    kptlabels = list(kpts)
    kpts = list(kpts.values())

    # Initialize strains and deformation potentials results
    strains = [-abs(strain), abs(strain)]
    defpots = {kpt: OrderedDict() for kpt in kptlabels}

    # Navigate the directories containing the ground states of
    # the strained structures and extract the band edges
    for ij in get_relevant_strains(pbc):
        straincomp = ijlabels[ij]
        edges_ij = []
        for strain in strains:
            folder = get_strained_folder_name(strain, ij[0], ij[1])
            edges = get_edges(folder, kpts, soc)
            edges_ij.append(edges)

        # Actual calculation of the deformation potentials
        defpots_ij = np.squeeze(
            np.diff(edges_ij, axis=0) / (np.ptp(strains) * 0.01)
        )

        for dp, kpt in zip(defpots_ij, kptlabels):
            defpots[kpt][straincomp] = {
                'VB': dp[0],
                'CB': dp[1]
            }

    return defpots


@command('asr.deformationpotentials',
         returns=DefPotsResult)
@option('-s', '--strain', type=float,
        help='percent strain applied to the material along all components')
def main(strain: float = 1.0) -> DefPotsResult:
    """Calculate deformation potentials.

    Calculate the deformation potentials both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary. The dictionary has the following structure:

    {'defpots_soc': {'kpt_1': {'xx': {'CB': <value>,
                                                     'VB': <value>},

                                              'yy': {...},
                                              'xy': {...}}

                                    'kpt_2': {...},
                                      ...
                                    'kpt_N': {...},
                                    'VBM': {...},
                                    'CBM': {...},
                                    'gap': {...}}

     'defpots_nosoc': ...}

     The coordinates of the k-points used with and withous SOC are included
     in the results as well.

    Parameters
    ----------
    strain: float
        Percent strain to apply, in both direction and for all
        the relevant strain components, to the current material.
    """
    from gpaw import GPAW

    calc = GPAW('gs.gpw')

    soclabels = {'defpots_nosoc': False,
                 'defpots_soc': True}

    results = {}
    for label, soc in soclabels.items():
        kpoints = get_relevant_kpts(calc, '.', soc)
        results[f'kpts_{label}'] = kpoints
        defpots = _main(calc.atoms.pbc, kpoints, gpaw_get_edges, strain, soc)
        results[label] = defpots

        # Updating results with band gap deformation potentials
        dp_vbm = get_table_row('VBM', 'VB', defpots)
        dp_cbm = get_table_row('CBM', 'CB', defpots)
        dp_gap = dp_cbm - dp_vbm
        results[label]['gap'] = {
            key: comp for key, comp in zip(['xx', 'yy', 'xy'], dp_gap)
        }

    return results


if __name__ == '__main__':
    main.cli()
