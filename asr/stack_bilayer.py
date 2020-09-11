from asr.core import command, option, AtomsFile
from ase import Atoms
import numpy as np


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
    import numpy as np
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
    import numpy as np

    pure_cell = Atoms('C', positions=[[0, 0, 0]])
    pure_cell.set_cell(atoms.get_cell())

    symmetryC = spglib.get_symmetry(pure_cell)

    rotations = symmetryC['rotations']
    translations = symmetryC['translations']

    final_mats = []
    labels = []
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
        inversion = '-Iz' if np.allclose(U_cc[2, 2], -1) else ''
        label = f'{U_cc[0, 0]}_{U_cc[0, 1]}_{U_cc[1, 0]}_{U_cc[1, 1]}' + inversion
        # labels.append(f'{str(atoms.symbols)}-{str(i)}')
        labels.append(f'{str(atoms.get_chemical_formula())}-2-' + label)
        transforms.append((U_cc, t_c))

    auxs = list(zip(labels, transforms))
    final_mats, auxs = unique_materials(final_mats, auxs)
    labels, transforms = zip(*auxs)

    return final_mats, labels, transforms


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
    import numpy as np
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


def append_material(x, y, mat, atoms, tform, label, labelsuffix,
                    layers, final_mats, translations,
                    final_labels, final_transforms):
    d = 12  # The arbitrary distance between layers in the prototype
    layered1 = translation(x, y, d, mat, atoms)
    layers.append(layered1)
    translations.append((x, y))
    final_mats.append(mat.copy())
    final_labels.append(label + '-' + labelsuffix)
    final_transforms.append(tform)


def build_layers(atoms, cell_type, rotated_mats, labels, transforms, rmsd_tol):
    base_positions = flatten(atoms, rmsd_tol)
    cell = atoms.cell

    full_labels = []
    bilayers = []
    toplayers = []
    symmetries = []
    translations = []
    for toplayer, label, (U_cc, t_c) in zip(rotated_mats, labels, transforms):
        top_positions = flatten(toplayer, rmsd_tol)

        for pos1 in base_positions:
            for pos2 in top_positions:
                top = toplayer.copy()
                move = pos1 - pos2
                move_c = np.array([move[0], move[1], 0.0])
                move_c = cell.scaled_positions(move_c)
                total_translation = move_c + t_c
                translation_label = pretty_float(total_translation)
                full_label = label + "-" + translation_label

                bilayer = translation(move[0], move[1], 12, toplayer, atoms)

                full_labels.append(full_label)
                bilayers.append(bilayer)
                toplayers.append(top)
                symmetries.append((U_cc, t_c))
                translations.append(move)

    _labels, _bis, _tops, _syms, _ts = cell_specific_stacks(atoms, cell_type,
                                                            rotated_mats, labels,
                                                            transforms, rmsd_tol)

    full_labels.extend(_labels)
    bilayers.extend(_bis)
    toplayers.extend(_tops)
    symmetries.extend(_syms)
    translations.extend(_ts)

    auxs = list(zip(toplayers, full_labels, translations, symmetries))
    unique_layers, unique_auxs = unique_materials(bilayers, auxs, full=True, rmsd_tol=rmsd_tol)
    tops, labels, translations, syms = zip(*unique_auxs)

    return tops, labels, translations, syms, unique_layers


def cell_specific_stacks(atoms, cell_type, rotated_mats, labels, transforms, rmsd_tol):
    full_labels = []
    bilayers = []
    toplayers = []
    symmetries = []
    final_transforms = []

    positions = atoms.get_positions()
    cell = atoms.cell
    unit_cell = atoms.get_cell_lengths_and_angles()
    a, b, c = unit_cell[:3]

    def append_helper(x, y, top, atoms, label, symtup):
        bilayer = translation(x, y, 12, top, atoms)
        
        move_c = np.array([x, y, 0.0])
        move_c = cell.scaled_positions(move_c)
        total_translation = move_c + t_c
        translation_label = pretty_float(total_translation)
        flabel = label + '-' + translation_label
        
        full_labels.append(flabel)
        bilayers.append(bilayer)
        toplayers.append(top.copy())
        symmetries.append(symtup)
        final_transforms.append(np.array([x, y]))
    
    if cell_type == 'hexagonal':
        for top, lab, (U_cc, t_c) in zip(rotated_mats, labels, transforms):
            x = positions[1, 0]
            y = positions[1, 1]
            append_helper(x, y, top, atoms, lab, (U_cc, t_c))

            x = 2 * positions[1, 0]
            y = 2 * positions[1, 1]
            append_helper(x, y, top, atoms, lab, (U_cc, t_c))
    elif cell_type in ['oblique', 'rectangular', 'square', 'centered']:
        for top, lab, stup in zip(rotated_mats, labels, transforms):
            x = a / 2.0
            y = 0.0
            append_helper(x, y, top, atoms, lab, stup)

            x = 0.0
            y = b / 2.0
            append_helper(x, y, top, atoms, lab, stup)

            x = a / 2.0
            y = b / 2.0
            append_helper(x, y, top, atoms, lab, stup)

    return full_labels, bilayers, toplayers, symmetries, final_transforms

def pretty_float(arr):
    f1 = round(arr[0], 2)
    if np.allclose(f1, 0.0):
        s1 = "0"
    else:
        s1 = str(f1)
    f2 = round(arr[1], 2)
    if np.allclose(f2, 0.0):
        s2 = "0"
    else:
        s2 = str(f2)
    
    return f'{s1}_{s2}'


def translation(x, y, z, rotated, base):
    stacked = base.copy()
    rotated = rotated.copy()
    rotated.translate([x, y, z])
    stacked += rotated
    stacked.wrap()

    return stacked
# def webpanel


@command(module='asr.stack_bilayer', requires=['structure.json'])
@option('-a', '--atoms', help='Monolayer to be stacked',
        type=AtomsFile(), default='structure.json')
@option('-t', '--rmsd-tol', help='Position comparison tolerance',
        default=0.3)
def main(atoms: Atoms,
         rmsd_tol):
    if sum(atoms.pbc) != 2:
        raise StackingError('It is only possible to stack 2D materials')
    import os

    from asr.core import write_json
    # Increase z vacuum
    atoms.cell[2, 2] *= 2
    # Center atoms
    spos_av = atoms.get_positions()
    spos_av[:, 2] += atoms.cell[2, 2] / 4.0
    atoms.set_positions(spos_av)

    cell_type = get_cell_type(atoms)

    rotated_mats, labels, transforms = get_rotated_mats(atoms)

    things = build_layers(atoms, cell_type,
                          rotated_mats,
                          labels, transforms, rmsd_tol)

    for mat, label, transl, tform, proto in zip(*things):
        if not os.path.isdir(label):
            os.mkdir(label)
        mat.cell[2, 2] /= 2
        spos_av = mat.get_positions()
        spos_av[:, 2] -= mat.cell[2, 2] / 2
        mat.set_positions(spos_av)
        mat.write(f'{label}/toplayer.json')
        proto.write(f'{label}/bilayerprototype.json')
        dct = {'translation_vector': transl}
        write_json(f'{label}/translation.json', dct)
        transform_data = {'rotation': tform[0],
                          'translation': tform[1]}
        write_json(f'{label}/transformdata.json', transform_data)

    return {'folders': labels}


if __name__ == '__main__':
    main.cli()
