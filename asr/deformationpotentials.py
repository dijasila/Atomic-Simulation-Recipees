"""Deformation potentials."""
import numpy as np
import typing

from asr.core import command, option, ASRResult, prepare_result
from asr.utils.gpw2eigs import calc2eigs
from asr.database.browser import href, make_panel_description


description_text = """\
The deformation potentials represent the energy shifts of the
bottom of the conduction band (CB) and the top of the valence band
(VB) at a given k-point, under an applied strain.

This panel shows the VB and CB deformation potentials at the
high-symmetry k-points, subdivided into the different strain
components. In case one or both the band extrema of the material
are not found at any of the special points, the corresponding
k-point(s) are added to the list as `VBM` and/or `CBM`
(indirect-gap materials) or `VBM/CBM` (direct-gap materials).

Results are calculated both with and without spin-orbit coupling (SOC).
If any significant difference (at least 10 meV) if found between
the two sets of deformation potentials, two separate tables are shown.
Otherwise, we report only the values calculated with SOC."""


panel_description = make_panel_description(
    description_text,
    articles=[
        href("""Wiktor, J. and Pasquarello, A., 2016. Absolute deformation potentials
of two-dimensional materials. Physical Review B, 94(24), p.245411""",
             "https://doi.org/10.1103/PhysRevB.94.245411")
    ],
)


def is_in_list(element, lst, tol):
    """Check if a list of floating-point numbers is contained in an array."""
    for obj in lst:
        if np.allclose(element, obj, rtol=tol, atol=0):
            return True
    return False


def get_relevant_kpts(atoms, calc):
    """Obtain the high-symmetry k-points.

    If the band edges of the unstrained material are found away
    from any of the special points, the corresponding
    k-points will be added to the list as 'VBM' and 'CBM'
    or 'VBM/CBM' (direct-gap materials)
    """
    from ase.dft.bandgap import bandgap

    # XXX investigate issue with special points falling outside cell vectors
    specpts = atoms.cell.bandpath(pbc=atoms.pbc, npoints=0).special_points

    _, ivbm, icbm = bandgap(calc, output=None)
    if ivbm[1] == icbm[1]:
        kdict = {'VBM/CBM': ivbm}
    else:
        kdict = {'VBM': ivbm,
                 'CBM': icbm}

    ibz_kpoints = calc.get_ibz_k_points()
    for label, i_kpt in kdict.items():
        kpt = ibz_kpoints[i_kpt[1]]
        if not is_in_list(kpt, specpts.values(), tol=0.01):
            specpts[label] = kpt

    return specpts


def webpanel(result, row, key_descriptions):
    from asr.database.browser import matrixtable, describe_entry, WebPanel

    description = describe_entry('Deformation potentials', panel_description)
    columnlabels = [
        'xx<sup>VB</sup>',
        'xx<sup>CB</sup>',
        'yy<sup>VB</sup>',
        'yy<sup>CB</sup>',
        'xy<sup>VB</sup>',
        'xy<sup>CB</sup>'
    ]

    labeldct = {'deformation_potentials_soc': 'SOC',
                'deformation_potentials_nosoc': 'NOSOC'}

    dp_arrays = []
    dp_tables = []
    for resultname, label in labeldct.items():
        data = result[resultname]
        dp_array = []
        rowlabels = []

        for kpt in data:
            if kpt == 'G':
                rowlabels.append('&#x0393')
            else:
                rowlabels.append(kpt)

            dp_array_row = []
            for comp in data[kpt]:
                for band in data[kpt][comp]:
                    dp_array_row.append(data[kpt][comp][band])
            dp_array.append(dp_array_row)
        dp_arrays.append(np.asarray(dp_array))

        dp_table = matrixtable(
            dp_array,
            digits=3,
            title=f'D<sup>{label}</sup> (eV)',
            columnlabels=columnlabels,
            rowlabels=rowlabels
        )
        dp_tables.append(dp_table)

    columns = [[dp_tables[0]]]
    diff = abs(dp_arrays[0] - dp_arrays[1])
    if np.any(diff > 0.05):
        columns.append(dp_tables[1])

    panel = WebPanel(
        description,
        columns=columns,
        sort=4
    )
    return [panel]


@prepare_result
class Result(ASRResult):

    deformation_potentials_nosoc: typing.Dict[str, float]
    deformation_potentials_soc: typing.Dict[str, float]
    kpts: typing.Union[list, typing.Dict[str, float]]

    key_descriptions = {
        'deformation_potentials_nosoc': (
            'Deformation potentials under different types of '
            'deformations (xx, yy, zz, yz, xz, xy) at each k-point, '
            'without SOC'),
        'deformation_potentials_soc': (
            'Deformation potentials under different applied strains '
            '(xx, yy, zz, yz, xz, xy) at each k-point, with SOC'),
        'kpts': 'k-points at which deformation potentials were calculated'
    }

    formats = {"ase_webpanel": webpanel}


soclabels = {'deformation_potentials_nosoc': False,
             'deformation_potentials_soc': True}

ijlabels = {
    (0, 0): 'xx',
    (1, 1): 'yy',
    (2, 2): 'zz',
    (0, 1): 'xy',
    (0, 2): 'xz',
    (1, 2): 'yz',
}


@command('asr.deformationpotentials',
         returns=Result)
@option('-s', '--strain', type=float,
        help='percent strain applied to the material along all components')
@option('--all-ibz', is_flag=True, type=bool,
        help=('Calculate deformation potentials at all '
              'the irreducible Brillouin zone k-points.'))
def main(strain: float = 1.0, all_ibz: bool = False) -> Result:
    """Calculate deformation potentials.

    Calculate the deformation potentials both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary. The dictionary has the following structure:

    {'deformation_potentials_soc': {'kpt_1': {'xx': {'CB': <value>,
                                                     'VB': <value>},

                                              'yy': {...},
                                              'xy': {...}}

                                    'kpt_2': {...},
                                      ...
                                    'kpt_N': {...}},

     'deformation_potentials_nosoc': ...}

    Parameters
    ----------
    strain-percent: float
        Percent strain to apply, in both direction and for all
        the relevant strain components, to the current material.
    all-ibz: bool
        If True, calculate the deformation potentials at all
        the k-points in the irreducible Brillouin zone.
        Otherwise, just use the special points and the k-points
        where the edge states are found (if they are not already
        at one of the special points).
    """
    from gpaw import GPAW
    from ase.io import read
    from asr.gs import vacuumlevels

    atoms = read('structure.json')
    calc = GPAW('gs.gpw')

    if all_ibz:
        kpts = calc.get_ibz_k_points()
    else:
        kpts = get_relevant_kpts(atoms, calc)

    def gpaw_get_edges(folder, kpts, soc):
        """Obtain the edge states at the different k-points.

        Returns, for each k-point included in the calculation,
        the top eigenvalue of the valence band and the bottom
        eigenvalue of the conduction band.
        """
        atoms = read(f'{folder}/structure.json')
        gpw = GPAW(f'{folder}/gs.gpw').fixed_density(
            kpts=kpts,
            symmetry='off',
            txt=None
        )
        gpw.get_potential_energy()
        all_eigs, efermi = calc2eigs(gpw, soc=soc)
        vac = vacuumlevels(atoms, calc)

        # This will take care of the spin polarization
        if not soc:
            all_eigs = np.hstack(all_eigs)

        edges = np.zeros((len(all_eigs), 2))
        for i, eigs_k in enumerate(all_eigs):
            vb = [eig for eig in eigs_k if eig - efermi < 0]
            cb = [eig for eig in eigs_k if eig - efermi > 0]
            edges[i, 0] = max(vb)
            edges[i, 1] = min(cb)
        return edges - vac.evacmean

    return _main(atoms.pbc, kpts, gpaw_get_edges, strain)


def _main(pbc, kpts, get_edges, strain):
    from asr.setup.strains import (get_relevant_strains,
                                   get_strained_folder_name)
    results = {
        'kpts': kpts
    }

    if isinstance(kpts, dict):
        kptlabels = list(kpts)
        kpts = list(kpts.values())
    else:
        kptlabels = kpts

    # Initialize strains and deformation potentials results
    strains = [-abs(strain), abs(strain)]
    results.update({socstr: {kpt: {} for kpt in kptlabels}
                    for socstr in soclabels})

    # Navigate the directories containing the ground states of
    # the strained structures and extract the band edges
    for socstr, soc in soclabels.items():
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
                results[socstr][kpt][straincomp] = {
                    'VB': dp[0],
                    'CB': dp[1]
                }

    return results


if __name__ == '__main__':
    main.cli()
