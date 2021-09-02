from gpaw.utilities.dipole import dipole_matrix_elements_from_calc
from asr.core import command, ASRResult, prepare_result
from asr.defect_symmetry import (return_defect_coordinates,
                                 check_and_return_input)
from gpaw import restart
from pathlib import Path
import numpy as np


@prepare_result
class Result(ASRResult):
    """Container for transition dipole moment results."""

    d_i: np.ndarray
    tdm: float

    key_descriptions = dict(
        d_i='x-, y-, and z-component of the transition dipole moment [Å].',
        tdm='Planar average of the transition dipole element [Å].')

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

    occ_spin0 = []
    ev_spin0 = calc.get_eigenvalues()
    [occ_spin0.append(en) for en in ev_spin0 if en < ef]

    n1 = len(occ_spin0)
    n2 = n1 + 1

    structure, unrelaxed, primitive, pristine = check_and_return_input()
    defectpath = Path('.')
    center = return_defect_coordinates(structure, unrelaxed, primitive, pristine,
                                       defectpath)
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
    d_nnv = d_snnv[0]
    element = [d_nnv[0, 0, 0], d_nnv[0, 0, 1], d_nnv[0, 0, 2]]
    DipMom = (element[0] + element[1]) / 2

    return Result.fromdata(
        d_i=element,
        tdm=DipMom)

    # print(f'Spin={spin}')
    # for direction, d_nn in zip('xyz', d_nnv.T):
    #     print(f' <{direction}>',
    #           ''.join(f'{n:8}' for n in range(n1, n2)))
    #     for n in range(n1, n2):
    #         print(f'{n:4}', ''.join(f'{d:8.3f}' for d in d_nn[n - n1]))



if __name__ == '__main__':
    main.cli()
