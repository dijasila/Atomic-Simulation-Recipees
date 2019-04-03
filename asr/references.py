# Creates: references.py

import click
"""References from OQMD.

See:

    https://cmr.fysik.dtu.dk/oqmd12/oqmd12.html
"""

from collections import Counter


# Code generated with main() function below.  Do not touch!
references = {
    'Kr': 0.029, 'N': -8.353, 'H': -3.324, 'F': -1.587, 'Si': -5.399,
    'Rh': -7.147, 'Ir': -9.276, 'V': -8.524, 'Ge': -4.516, 'Pt': -6.354,
    'Fe': -9.042, 'Al': -3.738, 'Na': -1.307, 'Ru': -9.463, 'Hg': -0.885,
    'Re': -11.658, 'Sb': -4.406, 'Li': -1.890, 'Y': -4.692, 'Zr': -7.403,
    'Zn': -1.184, 'Se': -3.483, 'C': -9.219, 'Rb': -0.909, 'O': -5.124,
    'Cd': -0.927, 'Co': -8.229, 'I': -1.478, 'As': -4.663, 'Be': -3.702,
    'Ca': -2.012, 'Au': -3.130, 'Sc': -4.660, 'Ag': -2.840, 'Cu': -3.676,
    'W': -11.485, 'Cl': -1.785, 'Xe': -0.086, 'Br': -1.585, 'Cr': -9.409,
    'Ta': -9.816, 'Ni': -7.247, 'In': -2.763, 'Sr': -1.674, 'Pb': -3.759,
    'Pd': -3.820, 'Mo': -11.202, 'Ga': -2.903, 'Nb': -10.326, 'Te': -3.205,
    'Ti': -6.690, 'Sn': -4.056, 'Os': -11.191, 'Hf': -7.506, 'Bi': -4.674,
    'Cs': -0.833, 'Tl': -2.498, 'K': -1.219, 'B': -6.705, 'Mg': -1.617,
    'Mn': -9.717, 'P': -5.365, 'S': -4.078, 'Ba': -1.828}
# end of computer generated code.


def formation_energy(atoms):
    return fenergy(atoms.get_potential_energy(),
                   Counter(atoms.get_chemical_symbols()))


def fenergy(energy, count):
    e0 = sum(n * references[symbol] for symbol, n in count.items())
    return energy - e0


@click.command()
@click.argument('path', type=str)
def main(path):
    from ase.db import connect
    db = connect(path)
    print('references = {', end='')
    for i, row in enumerate(db.select(ns=1, u=False)):
        if i % 5 == 0:
            print('\n   ', end='')
        print(" '{}': {:.3f},".format(row.symbols[0], row.energy / row.natoms),
              end='')
    print('}')


group = 'Property'

if __name__ == '__main__':
    main()
