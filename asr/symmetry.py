from gpaw import GPAW
from gpaw.spinorbit import get_symmetry_eigenvalues
from asr.core import command, option, ASRResult, prepare_result
from asr.database.browser import (entry_parameter_description,
                                  describe_entry, WebPanel)
from typing import List


def webpanel(result, row, key_descriptions):

    parameter_description = entry_parameter_description(
        row.data, 'asr.symmetry')
    description = ('Second order symmetry indicator of the occupied bands \n\n'
                   + parameter_description)

    datarow = [describe_entry('Symmetry indicator', description),
               result.indicators, result.Qc]

    summary = WebPanel(title='Summary',
                       columns=[[{'type': 'table',
                                  'header': ['Electronic properties',
                                             'Indicators', 'Qc'],
                                  'rows': [datarow]}]])

    return [summary]


@prepare_result
class Result(ASRResult):
    indicators: List[float]
    Qc: float
    key_descriptions = {'indicators': 'Higher order topological index',
                        'Qc': 'Fractionalized corner charges'}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.symmetry',
         requires=['high_sym.gpw'],
         dependencies=['asr.symmetry@calculate'],
         returns=Result)
@option('-so', '--spin_orbit', help='Toggle spin orbit coupling', type=bool)
def main(spin_orbit: bool = True) -> Result:
    # Not parallelized
    calc = GPAW('high_sym.gpw')
    atoms = calc.atoms
    kpts = atoms.cell.get_bravais_lattice(pbc=atoms.pbc).get_special_points()

    X = list(kpts.keys())
    kpts = list(kpts.values())
    # Set up parameters for symmetry eigenvalue calculation
    Nv = int(calc.get_number_of_electrons() / 2)
    r_v = [0, 0, 0]
    bands = range(0, Nv)
    sym = 'C3'

    P_i = []
    for ik in range(len(kpts)):
        ps = get_symmetry_eigenvalues(calc, symmetry=sym,
                                      ik=ik, spin_orbit=spin_orbit,
                                      bands=bands,
                                      symmetry_center=r_v,
                                      deg_tol=1.0e-3, eig_tol=1e-2)
        print(ps)
        if sum(ps) != (len(bands) * (spin_orbit + 1)):
            print('Warning eigenvalue sum does not obey constraints')
        P_i.append(ps)

    idxG = X.index('G')
    idxK = X.index('K')

    # for i, kpt in enumerate(kpts):
    #     if i == idxG:
    #         continue

    indicators = [P_i[idxK][0] - P_i[idxG][0],
                  P_i[idxK][1] - P_i[idxG][1],
                  P_i[idxK][2] - P_i[idxG][2]]

    Qc = 2 / 3 * (indicators[0] + indicators[1])
    return Result.fromdata(indicators=indicators, Qc=Qc)


@command(module='asr.symmetry')
def calculate():
    from pathlib import Path

    if not Path('high_sym.gpw').is_file():
        calc = GPAW('gs.gpw')
        atoms = calc.atoms
        kpts = atoms.cell.get_bravais_lattice(pbc=atoms.pbc).get_special_points()
        calc = calc.fixed_density(kpts=list(kpts.values()), txt='high_sym.txt',
                                  parallel={'domain': 1}, symmetry='off')

        calc.diagonalize_full_hamiltonian(nbands=30)
        calc.write('high_sym.gpw', mode='all')

    return
