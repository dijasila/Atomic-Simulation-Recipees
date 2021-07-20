from asr.core import command, ASRResult, prepare_result
from gpaw import restart
import numpy as np


def webpanel(result, row, key_description):
    from asr.database.browser import WebPanel, matrixtable

    zfs_array = np.zeros((2, 3))
    rowlabels = ['Spin 0', 'Spin 1']
    for i, element in enumerate(zfs_array):
        zfs_array[i, 0] = f"{result.zfs['D_vv'][i][0]:.2f}"
        zfs_array[i, 1] = f"{result.zfs['D_vv'][i][1]:.2f}"
        zfs_array[i, 2] = f"{result.zfs['D_vv'][i][2]:.2f}"

    zfs_table = matrixtable(zfs_array,
                            title='ZFS',
                            columnlabels=['D<sub>xx</sub>',
                                          'D<sub>yy</sub>',
                                          'D<sub>zz</sub>'],
                            rowlabels=rowlabels)

    zfs = WebPanel('HF coupling and zero-field-splitting',
                   columns=[[], [zfs_table]],
                   sort=2)

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
    from gpaw.zero_field_splitting import convert_tensor, zfs
    from ase.units import _e, _hplanck

    atoms, calc = restart('gs.gpw', txt=None)
    D_vv = zfs(calc)
    unit = 'MHz'
    scale = _e / _hplanck * 1e-6
    D, E, axis, D_vv = convert_tensor(D_vv * scale)

    print('D_ij = ('
          + ',\n        '.join('(' + ', '.join(f'{d:10.3f}' for d in D_v) + ')'
                               for D_v in D_vv)
          + ') ', unit)
    print('i, j = x, y, z')
    print(f'D = {D:.3f} {unit}')
    print(f'E = {E:.3f} {unit}')
    x, y, z = axis
    print(f'axis = ({x:.3f}, {y:.3f}, {z:.3f})')

    return Result.fromdata(
        D_vv=D_vv)
