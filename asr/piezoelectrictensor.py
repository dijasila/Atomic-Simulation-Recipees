from asr.core import command, option
from pathlib import Path


def webpanel(row, key_descriptions):


    if 'e_vvv' in row.data:
        def matrixtable(M, digits=2):
            table = M.tolist()
            shape = M.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = table[i][j]
                    table[i][j] = '{:.{}f}'.format(value, digits)
            return table

        e_ij = row.data.e_vvv[:, [0, 1, 2, 1, 0, 0],
                              [0, 1, 2, 2, 2, 1]]
        e0_ij = row.data.e0_vvv[:, [0, 1, 2, 1, 0, 0],
                                [0, 1, 2, 2, 2, 1]]

        etable = dict(
            header=['Piezoelectric tensor', '', ''],
            type='table',
            rows=matrixtable(e_ij))

        e0table = dict(
            header=['Clamped piezoelectric tensor', ''],
            type='table',
            rows=matrixtable(e0_ij))

        columns = [[etable, e0table], []]

        panel = [('Piezoelectric tensor', columns)]
    else:
        panel = ()
    things = ()
    return panel, things


@command()
@option('--strain-percent', help='Strain fraction.')
def main(strain_percent=1, kpts={'density': 6.0, 'gamma': False}):
    import numpy as np
    from gpaw import GPAW
    from ase.calculators.calculator import kptdensity2monkhorstpack
    from ase.units import Bohr
    from asr.core import read_json
    from asr.setup.strains import main as setupstrains
    from asr.setup.strains import get_relevant_strains, get_strained_folder_name

    if not setupstrains.done:
        setupstrains(strain_percent=strain_percent)

    # TODO: Clamped strains
    calc = GPAW('gs.gpw', txt=None)
    params = calc.parameters

    # Do not symmetrize the density
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True,
                          'time_reversal': False}

    # We need the eigenstates to a higher accuracy
    params['convergence']['density'] = 1e-8
    atoms = calc.atoms

    # From experience it is important to use
    # non-gamma centered grid when using symmetries.
    # Might have something to do with degeneracies, not sure.
    if 'density' in kpts:
        density = kpts.pop('density')
        kpts['size'] = kptdensity2monkhorstpack(atoms, density, True)
    params['kpts'] = kpts
    oldcell_cv = atoms.get_cell()
    vol = abs(np.linalg.det(oldcell_cv / Bohr))
    pbc_c = atoms.get_pbc()
    L = np.abs(np.linalg.det(oldcell_cv[~pbc_c][:, ~pbc_c] / Bohr))
    epsclamped_vvv = np.zeros((3, 3, 3), float)
    eps_vvv = np.zeros((3, 3, 3), float)

    ij = get_relevant_strains(atoms.pbc)

    for i, j in ij:
        phase_sc = np.zeros((2, 3), float)
        for s, sign in enumerate([-1, 1]):
            folder = get_strained_folder_name(sign * strain_percent, i, j)
            polresults = read_json(folder / 'results-asr.formalpolarization.json')
            phase_sc[s] = polresults['phase_c']

        dphase_c = phase_sc[1] - phase_sc[0]
        dphase_c -= np.round(dphase_c / (2 * np.pi)) * 2 * np.pi
        dphasedeps_c = dphase_c / (2 * strain_percent)
        eps_v = (-np.dot(dphasedeps_c, oldcell_cv / Bohr)
                 / (2 * np.pi * vol))
        if (~atoms.pbc).any():
            eps_v *= L

        eps_vvv[:, i, j] = eps_v
        eps_vvv[:, j, i] = eps_v

    data = {'eps_vvv': eps_vvv,
            'epsclamped_vvv': epsclamped_vvv}

    return data


if __name__ == '__main__':
    main.cli()
