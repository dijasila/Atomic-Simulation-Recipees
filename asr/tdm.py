from asr.core import ASRResult, prepare_result
from asr.defect_symmetry import (return_defect_coordinates,
                                 check_and_return_input,
                                 DefectInfo, WFCubeFile)
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
        d_snnv='transition dipole matrix elements.',
        n1='staterange minimum.',
        n2='staterange maximum.')


# @command(module='asr.tdm',
#         requires=['gs.gpw', 'structure.json'],
#         dependencies=['asr.gs@calculate'],
#         resources='1:1h',
#         returns=Result)
# @option('--primitivefile', help='Path to the primitive structure file.',
#        type=str)
# @option('--pristinefile', help='Path to the pristine supercell file'
#        '(needs to be of the same shape as structure.json).', type=str)
# @option('--unrelaxedfile', help='Path to an the unrelaxed '
#        'supercell file (only needed if --mapping is set).', type=str)
# @option('--defect', help='Specify whether analysis is conducted for a '
#        'defect system. Only works when asr.get_wfs ran before!', type=bool)
def main(primitivefile: str = 'primitive.json',
         pristinefile: str = 'pristine.json',
         unrelaxedfile: str = None,
         defect: bool = False) -> Result:
    """Calculate HOMO-LUMO transition dipole moment for a given structure."""
    from gpaw import GPAW
    from gpaw.utilities.dipole import dipole_matrix_elements_from_calc

    # collect relevant files and evaluate center of the defect structure
    structurefile = 'structure.json'
    if unrelaxedfile is None:
        unrelaxedfile = 'NO'
    structure, unrelaxed, primitive, pristine = check_and_return_input(
        structurefile, unrelaxedfile, primitivefile, pristinefile)
    calc = GPAW('gs.gpw', txt=None)

    # run fixed density calculation and get fermi level position
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})

    if defect:
        # evaluate position of the defect in the structure
        assert Path('results-asr.get_wfs.json').is_file(), (
            'you first need to run asr.get_wfs in this directory!')
        defectpath = Path('.')
        defectinfo = DefectInfo(defectpath=defectpath)
        center = return_defect_coordinates(structure, primitive, pristine, defectinfo)
        n1, n2 = get_state_numbers_from_defectpath(defectpath)
    else:
        center = get_center_of_cell(structure)
        n1, n2 = get_state_numbers_from_calc(calc)

    d_snnv = dipole_matrix_elements_from_calc(calc, n1, n2, center)

    return Result.fromdata(
        d_snnv=d_snnv,
        n1=n1,
        n2=n2)


def get_center_of_cell(atoms):
    """Return the center point of the cell of a given atoms object."""
    return 0.5 * atoms.cell.sum(axis=0)


def get_state_numbers_from_defectpath(path):
    # wfcubefiletemplate = WFCubeFile.cubefilenames()
    cubefilenames = list(path.glob('wf.*.cube'))
    n1, n2 = get_state_numbers_from_cubefilenames(cubefilenames)

    return n1, n2


def get_state_numbers_from_calc(calc):
    evs = calc.get_eigenvalues()
    ef = calc.get_fermi_level()
    occ = [ev for ev in evs if ev < ef]
    n1 = len(occ)
    n2 = n1 + 2

    return n1, n2


def get_state_numbers_from_cubefilenames(cubefilenames):
    assert len(cubefilenames) > 0, (
        'no wavefunction files available to evaluate TDM evaluation for gapstates!')
    numlist = []
    for filename in cubefilenames:
        wfcubefile = WFCubeFile.fromfilename(filename.name)
        num = wfcubefile.band
        numlist.append(num)

    n1 = min(numlist)
    n2 = max(numlist)
    # if there's only one state available from the cubefiles, then set n2 to n1 +2
    # (include higher states such that a transition dipole moment between different
    # states can be calculated)
    if n1 == n2:
        n2 = n1 + 2

    return n1, n2


if __name__ == '__main__':
    main.cli()
