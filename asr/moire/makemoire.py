import json
from typing import Union
import numpy as np
from pathlib import Path
from ase.db import connect
from ase import Atoms
from ase.io.jsonio import read_json
from ase.build import make_supercell
from asr.core import command, option, ASRResult, prepare_result
from asr.utils.moireutils import Bilayer
from asr.moire.findmoire import angle_between, get_atoms_and_stiffness


def get_parameters(solution, filename):
    dct = read_json(filename)
    uid_a = dct['uid_a']
    uid_b = dct['uid_b']
    info = dct[solution]
    coeffs_a = info['coeffs_a']
    coeffs_b = info['coeffs_b']
    twist = info['twist_angle']
    return uid_a, uid_b, coeffs_a, coeffs_b, twist


def stack_TMDs(atm_a, atm_b, stacking):

    def get_TMD_ordered_formula(atoms):
        metal_Z = (42, 74)
        chalc_Z = (16, 34, 52)
        symbols_a = np.asarray(atoms.get_chemical_symbols())
        metal_a = symbols_a[np.in1d(atoms.numbers, metal_Z)]
        chalc_a = symbols_a[np.in1d(atoms.numbers, chalc_Z)]
        return np.concatenate((metal_a, chalc_a))

    zeds_a = atm_a.get_scaled_positions()[:, 2]
    zeds_b = atm_b.get_scaled_positions()[:, 2]
    formula_a = get_TMD_ordered_formula(atm_a)
    formula_b = get_TMD_ordered_formula(atm_b)
    new_a = Atoms(formula_a, pbc=[True, True, False], cell=atm_a.cell)
    new_b = Atoms(formula_b, pbc=[True, True, False], cell=atm_b.cell)

    if stacking == "AA":
        pos_a = [[0, 0, zeds_a[0]],
                 [2 / 3, 1 / 3, zeds_a[1]],
                 [2 / 3, 1 / 3, zeds_a[2]]]
        pos_b = [[0, 0, zeds_b[0]],
                 [2 / 3, 1 / 3, zeds_b[1]],
                 [2 / 3, 1 / 3, zeds_b[2]]]

    if stacking == "AB":
        pos_a = [[0, 0, zeds_a[0]],
                 [2 / 3, 1 / 3, zeds_a[1]],
                 [2 / 3, 1 / 3, zeds_a[2]]]
        pos_b = [[1 / 3, 2 / 3, zeds_b[0]],
                 [0, 0, zeds_b[1]],
                 [0, 0, zeds_b[2]]]

    if stacking == "AAp":
        pos_a = [[0, 0, zeds_a[0]],
                 [2 / 3, 1 / 3, zeds_a[1]],
                 [2 / 3, 1 / 3, zeds_a[2]]]
        pos_b = [[2 / 3, 1 / 3, zeds_b[0]],
                 [0, 0, zeds_b[1]],
                 [0, 0, zeds_b[2]]]

    if stacking == "ABp":
        pos_a = [[2 / 3, 1 / 3, zeds_a[0]],
                 [0, 0, zeds_a[1]],
                 [0, 0, zeds_a[2]]]
        pos_b = [[1 / 3, 2 / 3, zeds_b[0]],
                 [0, 0, zeds_b[1]],
                 [0, 0, zeds_b[2]]]

    new_a.set_scaled_positions(pos_a)
    new_b.set_scaled_positions(pos_b)
    return new_a, new_b


def build_supercell(atoms, coeffs):
    '''Obtain 2D supercell for a single layer from a 
       linear combination of the unit cell vectors a and b

       e.g:
          initial_cell = [a, b, c]
          coeffs = [[1, 3], [3, 2]]
          new_cell = [a + 3b, 3a + 2b, c]
    '''
    T = [[coeffs[0][0], coeffs[0][1], 0],
         [coeffs[1][0], coeffs[1][1], 0],
         [0, 0, 1]]

    newcell = make_supercell(atoms, T, wrap=True)
    return newcell


def get_common_cell(atoms_a, atoms_b, stif_a, stif_b, twist):
    '''Returns the commmon cell that minimizes total strain,
       given the cells and the stiffness tensors of the materials.
    '''
    def rotate_vector(vector, alpha):
        cos = np.cos(alpha)[0]
        sin = np.sin(alpha)[0]
        rot_matrix = np.array([[cos, -sin],
                               [sin, cos]])
        return np.dot(rot_matrix, vector)

    atoms_a.rotate(twist, 'z', rotate_cell=True)
    ratio_ii, ratio_ij = np.split(stif_a / stif_b, [2], axis=0)
    ratio_ii += 1
    ratio_ij += 1
    cell_a = atoms_a.cell
    cell_b = atoms_b.cell
    celldiff = (cell_b - cell_a)[:2, :2]
    celldisp = celldiff / ratio_ii
    newcell_xy = cell_a[:2, :2] + celldisp
    angdiff_1 = angle_between(cell_a[0], cell_b[0])
    angdiff_2 = angle_between(cell_a[1], cell_b[1])
    new_a1 = rotate_vector(newcell_xy[0], angdiff_1 / ratio_ij)
    new_a2 = rotate_vector(newcell_xy[1], angdiff_2 / ratio_ij)

    return np.array([np.append(new_a1, 0),
                     np.append(new_a2, 0),
                     [0.0, 0.0, 1.0]])


def get_strain(cell_a, cell_b, commoncell):
    strain_a = np.dot(np.linalg.inv(cell_a), commoncell) - np.eye(3)
    strain_b = np.dot(np.linalg.inv(cell_b), commoncell) - np.eye(3)
    return strain_a, strain_b, \
        max(abs(strain_a).max() * 100, abs(strain_b).max() * 100)


'''
@prepare_result
class Result(ASRResult):

    uid_a: str
    uid_b: str
    stacking: str
    twist_angle: float
    maxstrain: float
    coeffs_a: typing.List[int]
    coeffs_b: typing.List[int]
    cell_a_original: typing.List[float]
    cell_b_original: typing.List[float]
    supercell_a: typing.List[float]
    supercell_b: typing.List[float]
    strain_a: typing.List[float]
    strain_b: typing.List[float]

    key_descriptions = {'uid_a': 'uid_a',
                        'uid_b': 'uid_b',
                        'stacking': 'stacking',
                        'twist_angle': 'twist_angle',
                        'maxstrain': 'maxstrain',
                        'coeffs_a': 'coeffs_a',
                        'coeffs_b': 'coeffs_b',
                        'cell_a_original': 'cell_a_original',
                        'cell_b_original': 'cell_b_original',
                        'supercell_a': 'supercell_a',
                        'supercell_b': 'supercell_b',
                        'strain_a': 'strain_a',
                        'strain_b': 'strain_a'}
'''


@command('asr.makemoire')
@option('-s', '--solution', type=int)
@option('--filename', type=str)
@option('--root', type=str)
@option('--make-subdirectory')
@option('--stacking')
@option('--database', type=str)
def main(solution,
         filename: str='initial.json',
         root: str='.',
         make_subdirectory: bool=False,
         stacking=None,
         database: str='/home/niflheim2/cmr/C2DB-ASR/collected-databases/c2db.db'):

    cellfile = f'{root}/cells.json'
    uid_a, uid_b, coeffs_a, coeffs_b, twist = get_parameters(solution, cellfile)
    atoms_a, atoms_b, stif_a, stif_b = get_atoms_and_stiffness(uid_a, uid_b, database)
    atoms_a.cell[2, 2] = 1.0
    atoms_b.cell[2, 2] = 1.0

    if stacking:
        atoms_a, atoms_b = stack_TMDs(atoms_a, atoms_b, stacking)

    atoms_a_sc = build_supercell(atoms_a, coeffs_a)
    atoms_b_sc = build_supercell(atoms_b, coeffs_b)
    atoms_a_sc.set_tags(1)
    atoms_b_sc.set_tags(0)
    new_cell = get_common_cell(atoms_a_sc, atoms_b_sc, stif_a, stif_b, twist)
    strain_a, strain_b, maxstrain = get_strain(
        atoms_a_sc.cell, atoms_b_sc.cell, new_cell)

    bilayer = Bilayer(atoms_a_sc + atoms_b_sc)
    bilayer.set_interlayer_distance(3.0)
    bilayer.set_vacuum(15.0)
    new_cell[2, 2] = bilayer.cell[2, 2]    # This is to correct a weird i/o bug
    bilayer.set_cell(new_cell, scale_atoms=True)

    assert np.allclose(bilayer.cell[:2, :2], new_cell[:2, :2])
    assert new_cell[2, 2] == bilayer.cell[2, 2]

    if make_subdirectory:
        direc = f'{root}/{len(bilayer)}_{twist:.1f}_{maxstrain:.2f}'
    else:
        direc = f'{root}'
    Path(direc).mkdir(exist_ok=True, parents=True)
    bilayer.write(f'{direc}/{filename}')

    results = {
        'uid_a': uid_a,
        'uid_b': uid_b,
        'cell_a_original': atoms_a.cell.tolist(),
        'cell_b_original': atoms_b.cell.tolist(),
        'coeffs_a': coeffs_a,
        'coeffs_b': coeffs_b,
        'twist_angle': twist,
        'stacking': stacking,
        'supercell_a': atoms_a_sc.cell.tolist(),
        'supercell_b': atoms_b_sc.cell.tolist(),
        'final_supercell': bilayer.cell.tolist(),
        'strain_a': strain_a.tolist(),
        'strain_b': strain_b.tolist(),
        'maxstrain': maxstrain
    }

    with open(f'{direc}/bilayer-info.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main.cli()
