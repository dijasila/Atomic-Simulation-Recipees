"""Effective Born charges."""
from asr.core import command, option, ASRResult, prepare_result
import numpy as np
import typing
from asr.database.browser import make_panel_description, href, describe_entry


reference = """\
M. N. Gjerding et al. Efficient Ab Initio Modeling of Dielectric Screening
in 2D van der Waals Materials: Including Phonons, Substrates, and Doping,
J. Phys. Chem. C 124 11609 (2020)"""


panel_description = make_panel_description(
    """The Born charge of an atom is defined as the derivative of the static
macroscopic polarization w.r.t. its displacements u_i (i=x,y,z). The
polarization in a periodic direction is calculated as an integral over Berry
phases. The polarization in a non-periodic direction is obtained by direct
evaluation of the first moment of the electron density. The Born charge is
obtained as a finite difference of the polarization for displaced atomic
configurations.  """,
    articles=[
        href(reference, 'https://doi.org/10.1021/acs.jpcc.0c01635')
    ]
)


def webpanel(result, row, key_descriptions):
    import numpy as np

    def matrixtable(M, digits=2, unit='', skiprow=0, skipcolumn=0):
        table = M.tolist()
        shape = M.shape

        for i in range(skiprow, shape[0]):
            for j in range(skipcolumn, shape[1]):
                value = table[i][j]
                table[i][j] = '{:.{}f}{}'.format(value, digits, unit)
        return table

    columns = [[], []]
    for a, Z_vv in enumerate(
            row.data['results-asr.borncharges.json']['Z_avv']):
        table = np.zeros((4, 4))
        table[1:, 1:] = Z_vv
        rows = matrixtable(table, skiprow=1, skipcolumn=1)
        sym = row.symbols[a]
        rows[0] = [f'Z<sup>{sym}</sup><sub>ij</sub>', 'u<sub>x</sub>',
                   'u<sub>y</sub>', 'u<sub>z</sub>']
        rows[1][0] = 'P<sub>x</sub>'
        rows[2][0] = 'P<sub>y</sub>'
        rows[3][0] = 'P<sub>z</sub>'

        for ir, tmprow in enumerate(rows):
            for ic, item in enumerate(tmprow):
                if ir == 0 or ic == 0:
                    rows[ir][ic] = '<b>' + rows[ir][ic] + '</b>'

        Ztable = dict(
            type='table',
            rows=rows)

        columns[a % 2].append(Ztable)

    panel = {'title': describe_entry('Born charges', panel_description),
             'columns': columns,
             'sort': 17}
    return [panel]


@prepare_result
class Result(ASRResult):

    Z_avv: np.ndarray
    sym_a: typing.List[str]

    key_descriptions = {'Z_avv': 'Array of borncharges.',
                        'sym_a': 'Chemical symbols.'}

    formats = {"ase_webpanel": webpanel}


@command('asr.borncharges',
         dependencies=['asr.gs@calculate'],
         requires=['gs.gpw'],
         returns=Result)
@option('--displacement', help='Atomic displacement (Å)', type=float)
@option('--tmp_gpw_folder',
        help='Folder where the temporary gpw file for each displacement is saved',
        type=str)
def main(displacement: float = 0.01, tmp_gpw_folder: str = 'in_displacement_folder') -> Result:
    """Calculate Born charges."""
    from gpaw import GPAW

    from ase.units import Bohr

    from asr.core import chdir, read_json
    from asr.formalpolarization import main as formalpolarization
    from asr.setup.displacements import main as setupdisplacements
    from asr.setup.displacements import get_all_displacements, get_displacement_folder

    if not setupdisplacements.done:
        setupdisplacements(displacement=displacement)

    calc = GPAW('gs.gpw', txt=None)
    atoms = calc.atoms
    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    Z_avv = []
    phase_ascv = np.zeros((len(atoms), 2, 3, 3), float)

    for ia, iv, sign in get_all_displacements(atoms):
        folder = get_displacement_folder(ia, iv, sign, displacement)

        with chdir(folder):
            if not formalpolarization.done:
                if tmp_gpw_folder == 'in_displacement_folder':
                    formalpolarization()
                else:
                    formalpolarization(gpwname=tmp_gpw_folder
                                       + folder.name + '_formalpol.gpw')

        polresults = read_json(folder / 'results-asr.formalpolarization.json')
        phase_c = polresults['phase_c']
        isign = [None, 1, 0][sign]
        phase_ascv[ia, isign, :, iv] = phase_c

    for phase_scv in phase_ascv:
        dphase_cv = (phase_scv[1] - phase_scv[0])
        mod_cv = np.round(dphase_cv / (2 * np.pi)) * 2 * np.pi
        dphase_cv -= mod_cv
        phase_scv[1] -= mod_cv
        dP_vv = (np.dot(dphase_cv.T, cell_cv).T
                 / (2 * np.pi * vol))
        Z_vv = dP_vv * vol / (2 * displacement / Bohr)
        Z_avv.append(Z_vv)

    Z_avv = np.array(Z_avv)
    data = {'Z_avv': Z_avv, 'sym_a': sym_a}

    return data


if __name__ == '__main__':
    main.cli()
