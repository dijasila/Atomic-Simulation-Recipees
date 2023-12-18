"""Generate strained atomic structures."""
from asr.core import command, option, ASRResult


def get_relevant_strains(pbc):
    import numpy as np
    if np.sum(pbc) == 3:
        ij = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    elif np.sum(pbc) == 2:
        ij = ((0, 0), (1, 1), (0, 1))
    elif np.sum(pbc) == 1:
        ij = ((2, 2), )
    return ij


def get_strained_folder_name(strain_percent, i, j, clamped=False):
    from pathlib import Path
    import numpy as np
    if strain_percent == 0:
        return Path('.')
    itov_i = ['x', 'y', 'z']
    name = itov_i[i] + itov_i[j]
    sign = ['', '+', '-'][int(np.sign(strain_percent))]
    strain_percent = abs(float(strain_percent))

    if clamped:
        return Path(f'strains-{sign}{strain_percent}%-{name}-clamped/')

    return Path(f'strains-{sign}{strain_percent}%-{name}/')


def setup_strains(strain_percent=1, kptdensity=6.0, copyparams=True, clamp_atoms=False):
    from ase.io import read
    from ase.parallel import world
    from pathlib import Path
    import numpy as np
    from asr.setup.params import main as setup_params
    from asr.core import chdir, read_json, write_json
    from ase.calculators.calculator import kpts2sizeandoffsets

    atoms = read('structure.json')
    ij = get_relevant_strains(atoms.pbc)
    cell_cv = atoms.get_cell()
    size, _ = kpts2sizeandoffsets(density=kptdensity, atoms=atoms)

    nk1, nk2, nk3 = size
    for i, j in ij:
        for sign in [-1, 1]:
            signed_strain = sign * strain_percent
            strain_vv = np.eye(3)
            strain_vv[i, j] += signed_strain / 100.0
            strain_vv = (strain_vv + strain_vv.T) / 2
            strained_cell_cv = np.dot(cell_cv, strain_vv)
            atoms.set_cell(strained_cell_cv, scale_atoms=True)
            folder = get_strained_folder_name(signed_strain, i, j, clamped=clamp_atoms)
            if world.rank == 0:
                folder.mkdir()
            if clamp_atoms:
                filename = str(folder / 'structure.json')
            else:
                filename = str(folder / 'unrelaxed.json')
            atoms.write(filename)
            if copyparams:
                paramsfile = Path('params.json')
                if paramsfile.is_file():
                    write_json(folder / 'params.json', read_json(paramsfile))

            with chdir(folder):
                params = {
                    'asr.relax': {
                        'calculator': {
                            'kpts': {
                                'size': size,
                                'gamma': True,
                            },
                            None: None,
                        },
                        'fixcell': True,
                        'allow_symmetry_breaking': True,
                        'fmax': 0.0001,
                    }
                }
                setup_params(params=params)


@command('asr.setup.strains')
@option('--strain-percent', help='Strain percentage', type=float)
@option('--kptdensity', help='Setup up relax and gs calc with fixed density',
        type=float)
@option('--copyparams/--dontcopyparams',
        help='Copy params.json from current folder into strained folders',
        is_flag=True)
def clamped(strain_percent: float = 1, kptdensity: float = 6.0,
            copyparams: bool = True) -> ASRResult:
    results = setup_strains(strain_percent=strain_percent, kptdensity=kptdensity,
                            copyparams=copyparams, clamp_atoms=True)
    return results


@command('asr.setup.strains')
@option('--strain-percent', help='Strain percentage', type=float)
@option('--kptdensity', help='Setup up relax and gs calc with fixed density',
        type=float)
@option('--copyparams/--dontcopyparams',
        help='Copy params.json from current folder into strained folders',
        is_flag=True)
def main(strain_percent: float = 1, kptdensity: float = 6.0,
         copyparams: bool = True) -> ASRResult:
    results = setup_strains(strain_percent=strain_percent, kptdensity=kptdensity,
                            copyparams=copyparams, clamp_atoms=False)
    return results


if __name__ == '__main__':
    main.cli()
