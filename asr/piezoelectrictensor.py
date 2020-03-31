from asr.core import command, option


def webpanel(row, key_descriptions):
    def matrixtable(M, digits=2):
        table = M.tolist()
        shape = M.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                value = table[i][j]
                table[i][j] = '{:.{}f}'.format(value, digits)
        return table

    piezodata = row.data['results-asr.piezoelectrictensor.json']
    e_vvv = piezodata['eps_vvv']
    e0_vvv = piezodata['eps_clamped_vvv']

    e_ij = e_vvv[:,
                 [0, 1, 2, 1, 0, 0],
                 [0, 1, 2, 2, 2, 1]]
    e0_ij = e0_vvv[:,
                   [0, 1, 2, 1, 0, 0],
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

    panel = {'title': 'Piezoelectric tensor',
             'columns': columns}

    return [panel]


@command(module="asr.piezoelectrictensor",
         dependencies=['asr.gs@calculate'],
         requires=['gs.gpw'],
         webpanel=webpanel)
@option('--strain-percent', help='Strain fraction.')
@option('--kpts', help='K-point dict for ES calculation.')
def main(strain_percent=1, kpts={'density': 6.0, 'gamma': False}):
    import numpy as np
    from gpaw import GPAW
    from ase.calculators.calculator import kptdensity2monkhorstpack
    from ase.units import Bohr
    from asr.core import read_json, chdir
    from asr.formalpolarization import main as formalpolarization
    from asr.relax import main as relax
    from asr.setup.strains import main as setupstrains
    from asr.setup.strains import clamped as setupclampedstrains
    from asr.setup.strains import get_relevant_strains, get_strained_folder_name

    if not setupstrains.done:
        setupstrains(strain_percent=strain_percent)

    if not setupclampedstrains.done:
        setupclampedstrains(strain_percent=strain_percent)

    # TODO: Clamped strains
    # TODO: converge density and states
    calc = GPAW('gs.gpw', txt=None)
    atoms = calc.atoms

    # From experience it is important to use
    # non-gamma centered grid when using symmetries.
    # Might have something to do with degeneracies, not sure.
    if 'density' in kpts:
        density = kpts.pop('density')
        kpts['size'] = kptdensity2monkhorstpack(atoms, density, True)

    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    pbc_c = atoms.get_pbc()
    if not all(pbc_c):
        N = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
    else:
        N = 1.0
    eps_clamped_vvv = np.zeros((3, 3, 3), float)
    eps_vvv = np.zeros((3, 3, 3), float)
    ij = get_relevant_strains(atoms.pbc)

    for clamped in [True, False]:
        for i, j in ij:
            phase_sc = np.zeros((2, 3), float)
            for s, sign in enumerate([-1, 1]):
                folder = get_strained_folder_name(sign * strain_percent, i, j,
                                                  clamped=clamped)
                with chdir(folder):
                    if not clamped and not relax.done:
                        relax()
                    if not formalpolarization.done:
                        formalpolarization(kpts=kpts)

                polresults = read_json(folder / 'results-asr.formalpolarization.json')
                phase_sc[s] = polresults['phase_c']

            dphase_c = phase_sc[1] - phase_sc[0]
            dphase_c -= np.round(dphase_c / (2 * np.pi)) * 2 * np.pi
            dphasedeps_c = dphase_c / (2 * strain_percent * 0.01)
            eps_v = (-np.dot(dphasedeps_c, cell_cv)
                     / (2 * np.pi * vol))
            eps_v *= N

            if clamped:
                epsref_vvv = eps_clamped_vvv
            else:
                epsref_vvv = eps_vvv

            epsref_vvv[:, i, j] = eps_v
            epsref_vvv[:, j, i] = eps_v

    data = {'eps_vvv': eps_vvv,
            'eps_clamped_vvv': eps_clamped_vvv}

    return data


if __name__ == '__main__':
    main.cli()
