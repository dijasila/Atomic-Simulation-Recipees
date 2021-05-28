from asr.core import command, option, AtomsFile, ASRResult, prepare_result
from ase import Atoms
import numpy as np
from asr.utils.bilayerutils import translation, layername
from typing import List


class StackingError(ValueError):
    pass


def flatten(atoms, tol):
    flats = []
    for atom in atoms:
        pos = atom.position[:2]
        if any(np.allclose(x, pos, tol) for x in flats):
            continue
        else:
            flats.append(pos)

    return flats


def get_cell_type(atoms):
    ac = np.allclose
    uc = atoms.get_cell_lengths_and_angles()

    if ac(uc[0], uc[1]) and ac(uc[5], 90):
        uc_type = 'square'
    elif round(uc[3]) == 90 and ac(uc[5], 90):
        uc_type = 'rectangular'
    elif ac(uc[0], uc[1]) and (ac(120, uc[5]) or ac(60, uc[5])):
        uc_type = 'hexagonal'
    elif (not ac(uc[0], uc[1])
          and not ac(uc[5], 90)
          and ac(uc[0] / 2, abs(uc[1] * np.cos(uc[5] * np.pi / 180)))):
        uc_type = 'centered'
    else:
        uc_type = 'oblique'

    return uc_type


def get_rotated_mats(atoms: Atoms):
    import spglib

    pure_cell = Atoms('C', positions=[[0, 0, 0]])
    pure_cell.set_cell(atoms.get_cell())

    symmetryC = spglib.get_symmetry(pure_cell)

    rotations = symmetryC['rotations']
    translations = symmetryC['translations']

    final_mats = []
    transforms = []

    for i, (U_cc, t_c) in enumerate(zip(rotations, translations)):
        rotated_atoms = atoms.copy()

        # Calculate rotated and translated atoms
        spos_ac = rotated_atoms.get_scaled_positions()
        spos_ac = np.dot(spos_ac, U_cc.T) + t_c

        # Move atoms
        rotated_atoms.set_scaled_positions(spos_ac)

        # Wrap atoms outside of unit cell back
        rotated_atoms.wrap(pbc=[1, 1, 1])

        final_mats.append(rotated_atoms)
        transforms.append((U_cc, t_c))

    final_mats, transforms = unique_materials(final_mats, transforms)

    return final_mats, transforms


def unique_materials(mats, auxs, full=False, rmsd_tol=None):
    unique_mats = []
    unique_auxs = []
    for (mat, aux) in zip(mats, auxs):
        if not any(atomseq(mat, x, full=full, rmsd_tol=rmsd_tol) for x in unique_mats):
            unique_mats.append(mat)
            unique_auxs.append(aux)

    return unique_mats, unique_auxs


def atomseq(atoms1, atoms2, full=False, rmsd_tol=None):
    """Check equivalence of atoms1 and 2.

    Take two materials and go through each atom
    to check if the materials are equivalent.

    The full option should only be used on stacked materials
    as it will incorrectly identify two monolayers as being
    equal even if the corresponding bilayers would be different.
    """
    from asr.database.rmsd import get_rmsd

    identical = same_positions(atoms1, atoms2)
    if full and not identical:
        # Can return None for very different structures
        rmsd = get_rmsd(atoms1.copy(), atoms2.copy()) or 1
        identical = identical or rmsd < rmsd_tol

    return identical


def same_positions(atoms1, atoms2):
    poss1 = atoms1.get_positions()
    poss2 = atoms2.get_positions()

    numbers1 = atoms1.get_atomic_numbers()
    numbers2 = atoms2.get_atomic_numbers()

    for (pos1, n1) in zip(poss1, numbers1):
        # Is there exactly one atom in atoms2
        # that is at same position and has same
        # atomic number?

        count = 0
        for (pos2, n2) in zip(poss2, numbers2):
            if n1 != n2:
                continue
            if np.linalg.norm(pos1 - pos2) > 10e-5:
                continue
            count += 1
            if count >= 2:
                raise ValueError('Two atoms are on top of each other')

        if count == 0:
            return False

    return True


def build_layers(atoms, cell_type, rotated_mats, transforms, rmsd_tol):
    base_positions = flatten(atoms, rmsd_tol)

    bilayers = []
    toplayers = []
    symmetries = []
    translations = []
    for toplayer, (U_cc, t_c) in zip(rotated_mats, transforms):
        top_positions = flatten(toplayer, rmsd_tol)

        for pos1 in base_positions:
            for pos2 in top_positions:
                top = toplayer.copy()
                move = pos1 - pos2

                bilayer = translation(move[0], move[1], 12, toplayer, atoms)

                bilayers.append(bilayer)
                toplayers.append(top)
                symmetries.append((U_cc, t_c))
                translations.append(move)

    _bis, _tops, _syms, _ts = cell_specific_stacks(atoms, cell_type,
                                                   rotated_mats,
                                                   transforms, rmsd_tol)

    bilayers.extend(_bis)
    toplayers.extend(_tops)
    symmetries.extend(_syms)
    translations.extend(_ts)

    auxs = list(zip(toplayers, translations, symmetries))
    unique_layers, unique_auxs = unique_materials(
        bilayers, auxs, full=True, rmsd_tol=rmsd_tol)
    tops, translations, syms = zip(*unique_auxs)

    return tops, translations, syms, unique_layers


def cell_specific_stacks(atoms, cell_type, rotated_mats, transforms, rmsd_tol):
    bilayers = []
    toplayers = []
    symmetries = []
    final_transforms = []

    positions = atoms.get_positions()
    a, b, c = atoms.cell.lengths()

    def append_helper(x, y, top, atoms, symtup):
        U_cc, t_c = symtup
        bilayer = translation(x, y, 12, top, atoms)

        bilayers.append(bilayer)
        toplayers.append(top.copy())
        symmetries.append(symtup)
        final_transforms.append(np.array([x, y]))

    if cell_type == 'hexagonal':
        for top, (U_cc, t_c) in zip(rotated_mats, transforms):
            if len(positions) == 1:
                break

            x = positions[1, 0]
            y = positions[1, 1]
            append_helper(x, y, top, atoms, (U_cc, t_c))

            x = 2 * positions[1, 0]
            y = 2 * positions[1, 1]
            append_helper(x, y, top, atoms, (U_cc, t_c))
    elif cell_type in ['oblique', 'rectangular', 'square', 'centered']:
        for top, stup in zip(rotated_mats, transforms):
            x = a / 2.0
            y = 0.0
            append_helper(x, y, top, atoms, stup)

            x = 0.0
            y = b / 2.0
            append_helper(x, y, top, atoms, stup)

            x = a / 2.0
            y = b / 2.0
            append_helper(x, y, top, atoms, stup)

    return bilayers, toplayers, symmetries, final_transforms


@prepare_result
class StackBilayerResult(ASRResult):
    folders: List[str]

    key_descriptions = dict(
        folders='Folders containing created bilayers')


@command(module='asr.stack_bilayer', requires=['structure.json'])
@option('-a', '--atoms', help='Monolayer to be stacked',
        type=AtomsFile(), default='structure.json')
@option('-t', '--rmsd-tol', type=float,
        help='Position comparison tolerance')
def main(atoms: Atoms,
         rmsd_tol: float = 0.3) -> ASRResult:
    from gpaw import mpi
    import os
    if sum(atoms.pbc) != 2:
        raise StackingError('It is only possible to stack 2D materials')
    if mpi.world.size != 1:
        raise ValueError('This recipe cannot be run in parallel')

    from asr.core import write_json
    # Increase z vacuum
    atoms.cell[2, 2] *= 2
    # Center atoms
    spos_av = atoms.get_positions()
    spos_av[:, 2] += atoms.cell[2, 2] / 4.0
    atoms.set_positions(spos_av)

    cell_type = get_cell_type(atoms)

    rotated_mats, transforms = get_rotated_mats(atoms)

    things = build_layers(atoms, cell_type,
                          rotated_mats,
                          transforms, rmsd_tol)

    names = []
    for mat, transl, tform, proto in zip(*things):
        # Unpack and transform data needed to construct bilayer name
        t = tform[1] + \
            atoms.cell.scaled_positions(np.array([transl[0], transl[1], 0.0]))
        name = layername(atoms.get_chemical_formula(), 2, tform[0], t)
        names.append(name)

        if not os.path.isdir(name):
            os.mkdir(name)

        mat.cell[2, 2] /= 2
        spos_av = mat.get_positions()
        spos_av[:, 2] -= mat.cell[2, 2] / 2
        mat.set_positions(spos_av)
        mat.write(f'{name}/toplayer.json')

        proto.write(f'{name}/bilayerprototype.json')

        dct = {'translation_vector': transl}
        write_json(f'{name}/translation.json', dct)

        transform_data = {'rotation': tform[0],
                          'translation': tform[1]}
        write_json(f'{name}/transformdata.json', transform_data)

    return StackBilayerResult.fromdata(folders=names)


if __name__ == '__main__':
    main.cli()
