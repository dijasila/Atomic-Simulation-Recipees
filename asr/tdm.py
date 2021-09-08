from gpaw.utilities.dipole import dipole_matrix_elements_from_calc
from asr.core import command, ASRResult, prepare_result
from asr.defect_symmetry import (return_defect_coordinates,
                                 check_and_return_input)
from gpaw import restart
from pathlib import Path
import numpy as np
import typing


@prepare_result
class Result(ASRResult):
    """Container for transition dipole moment results."""

    d_snnv: typing.List[np.ndarray]
    n1: int
    n2: int

    key_descriptions = dict(
        d_snnv='transition dipole matrix elements for both spin channels.',
        n1='staterange minimum.',
        n2='staterange maximum.')

    # formats = {'ase_webpanel': webpanel}


@command(module='asr.tdm',
         requires=['gs.gpw', 'structure.json'],
         dependencies=['asr.gs@calculate'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    """Calculate HOMO-LUMO transition dipole moment."""
    atoms, calc = restart('gs.gpw', txt=None)
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})
    ef = calc.get_fermi_level()

    n1, n2 = get_defect_state_limits()
    n1 = None
    n2 = None
    if n1 is None and n2 is None:
        occ_spin0 = []
        ev_spin0 = calc.get_eigenvalues()
        [occ_spin0.append(en) for en in ev_spin0 if en < ef]
        n1 = len(occ_spin0)
        n2 = n1 + 2

    print(f'INFO: max. and minimum state index: {n1}, {n2}.')

    structure, unrelaxed, primitive, pristine = check_and_return_input()
    defectpath = Path('.')
    center = return_defect_coordinates(structure, unrelaxed, primitive, pristine,
                                       defectpath)
    print(f'INFO: defect center at {center}.')
    # center = atoms.cell.sum(axis=0) / 2  # center of cell

    d_snnv = dipole_matrix_elements_from_calc(calc, n1, n2, center)

    if calc.wfs.world.rank == 0:
        d_nnv_0 = d_snnv[0]
        d_nnv_1 = d_snnv[1]
    else:
        d_nnv_0 = np.empty((1, 1, 3))
        d_nnv_1 = np.empty((1, 1, 3))
    calc.wfs.world.broadcast(d_nnv_0, 0)
    calc.wfs.world.broadcast(d_nnv_1, 0)
    d_snnv = [d_nnv_0, d_nnv_1]
    print(d_snnv)
    # d_nnv = d_snnv[0]
    # print(d_snnv)
    # element = [d_nnv[0, 0, 0], d_nnv[0, 0, 1], d_nnv[0, 0, 2]]
    # DipMom = (element[0] + element[1]) / 2
    print('INFO: save results.')

    return Result.fromdata(
        d_snnv=d_snnv,
        n1=n1,
        n2=n2)

    # print(f'Spin={spin}')
    # for direction, d_nn in zip('xyz', d_nnv.T):
    #     print(f' <{direction}>',
    #           ''.join(f'{n:8}' for n in range(n1, n2)))
    #     for n in range(n1, n2):
    #         print(f'{n:4}', ''.join(f'{d:8.3f}' for d in d_nn[n - n1]))


def get_defect_state_limits():
    p = Path('.')
    pathlist = list(p.glob('wf*.cube'))
    if len(pathlist) == 0:
        return None, None

    numlist = []
    for path in pathlist:
        num = int(str(path.absolute()).split('/')[-1].split('.')[1].split('_')[0])
        numlist.append(num)

    n1 = min(numlist)
    n2 = max(numlist)

    if n1 == n2:
        n2 = n1 + 2

    return n1, n2


if __name__ == '__main__':
    main.cli()
