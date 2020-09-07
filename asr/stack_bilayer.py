from asr.core import command, option, AtomsFile
from ase import Atoms
import numpy as np


class StackingError(ValueError):
    pass


def flatten(atoms):
    flats = []
    for atom in atoms:
        pos = atom.position[:2]
        if any(np.allclose(x, pos) for x in flats):
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
        label = f'({U_cc[0, 0]}, {U_cc[0, 1]}, {U_cc[1, 0]}, {U_cc[1, 1]})' + inversion
        # labels.append(f'{str(atoms.symbols)}-{str(i)}')
        labels.append(f'{str(atoms.symbols)}-2-' + label)
        transforms.append((U_cc, t_c))

    auxs = list(zip(labels, transforms))
    final_mats, auxs = unique_materials(final_mats, auxs)
    labels, transforms = zip(*auxs)

    return final_mats, labels, transforms


def unique_materials(mats, auxs, full=False):
    unique_mats = []
    unique_auxs = []
    for (mat, aux) in zip(mats, auxs):
        if not any(atomseq(mat, x, full=full) for x in unique_mats):
            unique_mats.append(mat)
            unique_auxs.append(aux)

    return unique_mats, unique_auxs


def atomseq(atoms1, atoms2, full=False):
    """Check equivalence of atoms1 and 2.

    Take two materials and go through each atom
    to check if the materials are equivalent.

    The full option should only be used on stacked materials
    as it will incorrectly identify two monolayers as being
    equal even if the corresponding bilayers would be different.
    """
    from asr.database.rmsd import get_rmsd

    identical = same_positions(atoms1, atoms2)
    if full:
        # Can return None for very different structures
        rmsd = get_rmsd(atoms1, atoms2) or 1
        identical = identical or rmsd < 1e-3

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


def build_layers(atoms, cell_type, rotated_mats, labels, transforms):
    base_positions = flatten(atoms)
    cell = atoms.cell

    full_labels = []
    bilayers = []
    toplayers = []
    symmetries = []
    translations = []
    for toplayer, label, (U_cc, t_c) in zip(rotated_mats, labels, transforms):
        top_positions = flatten(toplayer)
        print(f"# OF COMBOS {len(top_positions) * len(base_positions)}")

        for pos1 in base_positions:
            for pos2 in top_positions:
                move = pos1 - pos2
                print(move, move[0], move[1])
                move_c = np.array([move[0], move[1], 0.0])
                move_c = cell.scaled_positions(move_c)
                total_translation = move_c + t_c
                translation_label = pretty_float(total_translation)
                full_label = label + "-" + translation_label

                full_labels.append(full_label)
                bilayer = translation(move[0], move[1], 12, toplayer, atoms)
                bilayers.append(bilayer)

                toplayers.append(toplayer)

                symmetries.append((U_cc, t_c))

                translations.append(move)

    auxs = list(zip(toplayers, full_labels, translations, symmetries))
    unique_layers, unique_auxs = unique_materials(bilayers, auxs, full=True)
    tops, labels, translations, syms = zip(*unique_auxs)

    return tops, labels, translations, syms, unique_layers


def pretty_float(arr):
    return f'({str(round(arr[0], 2))}, {str(round(arr[1], 2))})'


def _build_layers(atoms, cell_type, rotated_mats, labels, transforms):
    unit_cell = atoms.get_cell_lengths_and_angles()

    layers = []
    translations = []
    final_mats = []
    final_labels = []
    final_transforms = []

    positions = atoms.get_positions()
    a, b, c = unit_cell[:3]

    if cell_type == 'hexagonal':
        for (mat, label, tform) in zip(rotated_mats,
                                       labels,
                                       transforms):
            # Appender stacks the layers and adds an additional
            # translation of xy and modifies the label by
            # adding a suffix.
            # Then the results are appended to appropriate lists.
            def appender(x, y, suff):
                append_material(x, y, mat, atoms, tform, label, suff, layers,
                                final_mats, translations, final_labels,
                                final_transforms)

            appender(0, 0, '00')

            x = positions[1, 0]
            y = positions[1, 1]
            appender(x, y, '11')

            x = positions[1, 0] * 2
            y = positions[1, 1] * 2
            appender(x, y, '22')

    elif cell_type in ['oblique', 'rectangular', 'square', 'centered']:
        for (mat, label, tform) in zip(rotated_mats,
                                       labels,
                                       transforms):
            # Appender stacks the layers and adds an additional
            # translation of xy and modifies the label by
            # adding a suffix.
            # Then the results are appended to appropriate lists.
            def appender(x, y, suff):
                append_material(x, y, mat, atoms, tform, label, suff, layers,
                                final_mats, translations, final_labels,
                                final_transforms)

            appender(0, 0, '00')

            x = a / 2.0
            y = 0.0
            appender(x, y, '10')

            x = 0.0
            y = b / 2.0
            appender(x, y, '01')

            x = a / 2.0
            y = b / 2.0
            appender(x, y, '11')
    else:
        raise ValueError(f'Invalid cell type: {cell_type}')

    auxs = list(zip(final_mats, final_labels,
                    translations, final_transforms))

    un_layers, un_auxs = unique_materials(layers, auxs, full=True)
    mats, labels, translations, tforms = zip(*un_auxs)

    return mats, labels, translations, tforms, un_layers


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
def main(atoms: Atoms):
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
                          labels, transforms)

    rotated_mats, labels, translations, transforms, protos = things

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
