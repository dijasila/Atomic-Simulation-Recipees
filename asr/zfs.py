from asr.core import command, ASRResult, prepare_result
from asr.webpages.browser import WebPanel
import numpy as np


def get_zfs_table(result):
    zfs_array = np.zeros((2, 3))
    rowlabels = ['Spin 0', 'Spin 1']
    for i, element in enumerate(zfs_array):
        for j in range(3):
            zfs_array[i, j] = result['D_vv'][i][j]

    rows = []
    for i in range(len(zfs_array)):
        rows.append((rowlabels[i],
                     f'{zfs_array[i][0]:.2f} MHz',
                     f'{zfs_array[i][1]:.2f} MHz',
                     f'{zfs_array[i][2]:.2f} MHz'))

    zfs_table = {'type': 'table',
                 'header': ['Spin channel',
                            'D<sub>xx</sub>',
                            'D<sub>yy</sub>',
                            'D<sub>zz</sub>']}
    zfs_table['rows'] = rows

    return zfs_table


def webpanel(result, row, key_description):
    zfs_table = get_zfs_table(result)
    zfs = WebPanel('Zero field splitting (ZFS)',
                   columns=[[], [zfs_table]],
                   sort=41)

    return [zfs]


@prepare_result
class Result(ASRResult):
    """Container for zero-field-splitting results."""

    D_vv: np.ndarray

    key_descriptions = dict(
        D_vv='Zero-field-splitting components for each spin channel '
             'and each direction (x, y, z) [MHz].')

    formats = {'ase_webpanel': webpanel}


@command(module='asr.zfs',
         requires=['gs.gpw', 'structure.json'],
         dependencies=['asr.gs'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    """Calculate zero-field-splitting."""
    from gpaw import restart

    # read in atoms and calculator, check whether spin and magnetic
    # moment are suitable, i.e. spin-polarized triplet systems
    atoms, calc = restart('gs.gpw', txt=None)
    magmom = atoms.get_magnetic_moment()
    _ = check_magmoms(magmom)

    # run fixed density calculation
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})

    # evaluate zero field splitting components for both spin channels
    D_vv = get_zfs_components(calc)

    return Result.fromdata(
        D_vv=D_vv)


def get_zfs_components(calc):
    """Get ZFS components from GPAW function, and convert tensor scale."""
    from ase.units import _e, _hplanck
    from gpaw.zero_field_splitting import convert_tensor, zfs

    D_vv = zfs(calc)
    scale = _e / _hplanck * 1e-6
    D, E, axis, D_vv = convert_tensor(D_vv * scale)

    return D_vv


def check_magmoms(magmom, target_magmom=2, threshold=1e-1):
    """Check whether input atoms are a triplet system."""
    if not abs(magmom - target_magmom) < threshold:
        raise ValueError('ZFS recipe only working for systems with '
                         f'a total magnetic moment of {target_magmom}!')


if __name__ == '__main__':
    main.cli()
