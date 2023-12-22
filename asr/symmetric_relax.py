"""
Finds the relaxed atomic structure with enforced symmetry.

The Algorithm proceeds as follows:
  - Finds all unique space groups of structure for any symmetry precision.
  - Runs relaxation of each with symmetry constrained forces and stresses
  - Determines the lowest energy structure
    (-) If multiple structures have the same energy when rounded to
        convergence criteria on the energy, chooses the one with most symmetries
"""

from typing import List
import numpy as np
from ase import Atoms
from ase.io import write

from asr.core import (ASRResult, AtomsFile, DictStr, command, option,
                      prepare_result)
import spglib
from ase.utils import atoms_to_spglib_cell
from ase.spacegroup.symmetrize import SpgError
from asr.relax import relax, update_gpaw_paramters


def spg_consistent_symprec(atoms, verbose=False):
    symmetries = []
    atoms_list = []
    for symprec in np.logspace(0, -5, 21):
        try:
            symmetric_atoms = SpgAtoms.from_atoms(atoms, symprec)
        except SpgError:
            continue

        if verbose:
            print(symprec, symmetric_atoms.symmetry)
        if symmetric_atoms.symmetry not in symmetries:
            symmetries.append(symmetric_atoms.symmetry)
            atoms_list.append(symmetric_atoms)
    return atoms_list


class SpgAtoms(Atoms):

    @classmethod
    def from_atoms(cls, atoms, symprec):
        from ase.spacegroup.symmetrize import refine_symmetry
        refined_atoms = refine_symmetry(atoms, symprec=symprec, verbose=False)
        symmetric_atoms = cls(refined_atoms)
        symmetric_atoms.init_symprec = symprec

        # In order for the symmetric atoms to have the target space group of the
        # refined structure, it must not have a new_symprec larger than the
        # refinement init_symprec.
        # We also cannot choose arbitrarily low symprec due to numerical noise.
        new_symprec = 1e-5

        # Extract symmetries of symmetrized atoms
        dataset = spglib.get_symmetry_dataset(atoms_to_spglib_cell(symmetric_atoms),
                                              symprec=new_symprec)
        symmetric_atoms.dataset = dataset
        symmetric_atoms.set_symmetries(dataset['translations'], dataset['rotations'],
                                       new_symprec)
        return symmetric_atoms

    @property
    def symmetry(self):
        return (self.dataset['number'], tuple(self.dataset['wyckoffs']),
                tuple(self.dataset['equivalent_atoms']))

    def set_symmetries(self, t_sc, op_scc, new_symprec):
        self.op_svv = [np.linalg.inv(self.cell).dot(op_cc.T).dot(self.cell) for
                       op_cc in op_scc]
        self.nsym = len(op_scc)
        spos_ac = self.get_scaled_positions()
        a_sa = []

        for op_cc, t_c in zip(op_scc, t_sc):
            symspos_ac = np.dot(spos_ac, op_cc.T) + t_c

            a_a = []
            for s_c in symspos_ac:
                diff_ac = spos_ac - s_c
                diff_ac -= np.round(diff_ac)
                mask_c = np.all(np.abs(diff_ac) < new_symprec, axis=1)
                assert np.sum(mask_c) == 1, f'Bad symmetry, {mask_c}'
                ind = np.argwhere(mask_c)[0][0]
                assert ind not in a_a, f'Bad symmetry {ind}, {diff_ac}'
                a_a.append(ind)
            a_sa.append(a_a)

        self.a_sa = np.array(a_sa)

    def get_stress(self, voigt=True, *args, **kwargs):
        sigma0_vv = Atoms.get_stress(self, voigt=False, *args, **kwargs)

        sigma_vv = np.zeros((3, 3))
        for op_vv in self.op_svv:
            sigma_vv += np.dot(np.dot(op_vv, sigma0_vv), op_vv.T)
        sigma_vv /= self.nsym

        if voigt:
            return sigma_vv.flat[[0, 4, 8, 5, 2, 1]]

        return sigma_vv

    def get_forces(self, *args, **kwargs):
        f0_av = Atoms.get_forces(self, *args, **kwargs)
        f_av = np.zeros_like(f0_av)
        for map_a, op_vv in zip(self.a_sa, self.op_svv):
            for a1, a2 in enumerate(map_a):
                f_av[a2] += np.dot(f0_av[a1], op_vv)
        f_av /= self.nsym
        return f_av


@prepare_result
class Result(ASRResult):
    """Result class for :py:func:`asr.relax.main`."""

    atoms_list: List[Atoms]
    etot_per_electron_list: List[float]
    nsym_list: List[int]
    key_descriptions = \
        {'atoms_list': 'List of symmetrically relaxed atomic structure',
         'etot_per_electron_list': 'Total energies of respective structures',
         'nsym_list': 'Number of symmetries of respective structures'}


@command('asr.symmetric_relax',
         creates=['symmetric_structure.json'],
         returns=Result)
@option('-a', '--atoms', help='Atoms to be relaxed.',
        type=AtomsFile(), default='structure.json')
@option('-c', '--calculator', help='Calculator and its parameters.',
        type=DictStr())
@option('--d3/--nod3', help='Relax with vdW D3.', is_flag=True)
@option('--fixcell/--dont-fixcell',
        help="Don't relax stresses.",
        is_flag=True)
@option('--fmax', help='Maximum force allowed.', type=float)
def main(atoms: Atoms = 'structure.json',
         calculator: dict = {'name': 'gpaw',
                             'mode': {'name': 'pw', 'ecut': 800},
                             'xc': 'PBE',
                             'kpts': {'density': 6.0, 'gamma': True},
                             'basis': 'dzp',
                             'symmetry': {'symmorphic': False},
                             'convergence': {'forces': 1e-4},
                             'txt': 'relax.txt',
                             'occupations': {'name': 'fermi-dirac',
                                             'width': 0.05},
                             'charge': 0},
         d3: bool = False,
         fixcell: bool = False,
         fmax: float = 0.01) -> Result:
    """Relax atomic positions and unit cell.

    The relaxed structure is saved to `structure.json` which can be processed
    by other recipes.

    Parameters
    ----------
    atoms
        Atomic structure to relax.
    calculator
        Calculator dictionary description.
    d3
        Relax using DFTD3.
    fixcell
        Fix cell when relaxing, thus only relaxing atomic positions.
    fmax
        Maximum force tolerance.
    """
    from ase.calculators.calculator import get_calculator_class

    calculatorname = calculator.pop('name')
    traj_file = calculator.get('txt', '-').replace('.txt', '.traj')
    txt = calculator.pop('txt', '-')
    Calculator = get_calculator_class(calculatorname)

    # Some calculator specific parameters
    if calculatorname == 'gpaw':
        calculator = update_gpaw_paramters(atoms, calculator)

    # Extract structures within large symmetry precision range.
    atoms_list = spg_consistent_symprec(atoms)
    symmetric_results = []
    for i, symmetric_atoms in enumerate(atoms_list):
        # Create file names for each structure with different space group
        itraj_file = str(i) + traj_file
        itxt = str(i) + txt

        # # Determine restart conditions, not parallel safe
        # if os.path.isfile(itraj_file):
        #     print('trajectory file exists:', itraj_file)
        #     restart_atoms = read(itraj_file)
        #     symmetric_atoms = SpgAtoms.from_atoms(restart_atoms,
        #                                           symmetric_atoms.init_symprec)
        #     open_mode = 'a'
        # else:
        #     print('trajectory file doesnt exist')
        #     open_mode = 'w'

        # Relax structure, saves atoms object and energies tuple in results list
        open_mode = 'w'
        symmetric_structure = relax(symmetric_atoms, calculator, d3,
                                    open_mode, itxt, fmax, Calculator,
                                    itraj_file, calculatorname, fixcell)
        symmetric_results.append(symmetric_structure)

    # Sort structure(s) with by energy / electron
    total_energies = [result[3] for result in symmetric_results]
    sorted_energies, sorted_results = zip(*sorted(zip(total_energies,
                                                      symmetric_results)))

    # Select lowest energy structures, degenerate under convergence criteria
    convergence = calculator.get('convergence').get('energy')
    if convergence is None:
        rounding = int(-np.log10(5e-4))  # Current GPAW default
    else:
        rounding = int(-np.log10(convergence))

    sorted_energies = np.round(sorted_energies, rounding)
    minimum_results = []
    for energy, result in zip(sorted_energies, sorted_results):
        if energy == sorted_energies[0]:
            minimum_results.append(result)

    # Choose the one with most symmetries
    number_of_symmetries = np.array([result[0].nsym for result in minimum_results])
    _, high_symmetry_results = zip(*sorted(zip(number_of_symmetries,
                                               minimum_results)))
    atoms, _, _, etot_per_electron = high_symmetry_results[-1]
    write('symmetric_structure.json', atoms)

    return Result.fromdata(
        atoms_list=[result[0].copy() for result in sorted_results],
        etot_per_electron_list=[result[3].copy() for result in sorted_results],
        nsym_list=[result[0].nsym for result in sorted_results]
    )


if __name__ == '__main__':
    main.cli()
