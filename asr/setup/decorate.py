"""Generate chemically similar atomic structures."""
import typing
from asr.core import command, option, atomsopt
import numpy as np
from pathlib import Path


def apply_substitution(atoms, substitution):
    from ase.data import covalent_radii
    new_atoms = atoms.copy()
    new_numbers = [substitution[number] for number in atoms.numbers]

    # Scale in-plane lattice vectors by the harmonic mean of the covalent_radii
    sf = (np.product([covalent_radii[n] for n in new_numbers])
          / np.product([covalent_radii[n] for n in atoms.numbers]))
    sf = pow(sf, 1. / len(new_numbers))
    sf = [sf if atoms.pbc[n] else 1 for n in range(3)]
    newcell = np.diag(sf).dot(new_atoms.get_cell())
    new_atoms.set_cell(newcell, scale_atoms=True)

    # update the atomic numbers
    new_atoms.numbers = new_numbers
    return new_atoms


def find_substitutions(number, data, threshold):
    from ase.data import atomic_numbers
    row = data[number]
    substitutions = np.where(row > threshold)[0]
    allowed_elements = [
        'H', 'He',
        'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',

        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',

        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        # 6
        'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Rn']
    allowed_numbers = [atomic_numbers[e] for e in allowed_elements]
    substitutions = [s for s in substitutions if s in allowed_numbers]
    return substitutions


def generate_structures(prototype, p_ab, threshold=0.08):
    from itertools import product
    numbers = list(set(prototype.numbers))
    substitutions = [find_substitutions(number, p_ab, threshold)
                     for number in numbers]
    for substitution in product(*substitutions):
        new_numbers = list(set(substitution))
        if len(new_numbers) != len(numbers):
            continue
        yield prototype, dict(zip(numbers, substitution))


def get_p_ab():
    """Get similarity matrix.

    The data is saved as a matrix of counts, so that s_ab gives the number of
    times that a can substitute for b in the icsd. This is be normalized,
    to give a probability of succesful substitution. See

    """
    name = Path(__file__).parent / 'substitution.dat'
    s_ab = np.loadtxt(name)
    from ase.utils import seterr
    with seterr(divide='ignore', invalid='ignore'):
        tmp = s_ab ** 2 / s_ab.sum(axis=0)[None, :] / s_ab.sum(axis=1)[:, None]
    tmp[np.isnan(tmp)] = 0
    np.fill_diagonal(tmp, 1)

    return np.sqrt(tmp)


@command('asr.setup.decorate')
@atomsopt
@option('--threshold',
        help='Threshold of likelyhood of two atomic species to subsititute',
        type=float)
def main(
        atoms: str,
        threshold: float = 0.08,
) -> typing.List:
    """Create similar atomic structures.

    This recipe can substitute atoms in an atomic structure with other similar
    atoms. In this case, similarity is defined as a probability describing the
    number of experimentally known atomic structures which only differ
    by a simple substitution, say Si -> Ge.

    The number of coexisting atomic structures has been analyzed in Ref. XXX
    and this recipe is converting this number to a probability.

    The threshold option limits the number of performed atomic substitions to
    the ones that have a probability larger than the threshold.

    """
    from ase.data import chemical_symbols
    p_ab = get_p_ab()

    decorated = []
    for structure, subs in generate_structures(atoms, p_ab,
                                               threshold=threshold):
        structure = apply_substitution(structure, subs)
        formula = structure.symbols.formula
        explanation = ''
        substitutions = []
        for i, j in subs.items():
            a = chemical_symbols[i]
            b = chemical_symbols[j]
            prob = p_ab[i, j]
            explanation += f'{a}->{b} (P={prob:.3f})'
            substitutions.append((a, b, prob))
        print(f'Created {formula:metal}: {explanation}')
        decoration = dict(atoms=structure, substitutions=substitutions)
        decorated.append(decoration)
    return decorated


if __name__ == "__main__":
    main.cli()
