from asr.utils import command, option


def phononbuildingblock(atoms, Z_avv, force_constant_matrix):
    # Simple function for calculating phonon building block

    phbb = {'Z_avv': Z_avv,
            'C_NN': force_constant_matrix,
            'masses': atoms.get_masses(),
            'symbols': atoms.get_chemical_symbols(),
            'cell': atoms.cell}

    return phbb


@command(module='asr.phononbuildingblock',
         requires=['borncharges-0.01.json',
                   'results-asr.phonons@calculate.json'],
         dependencies=['asr.phonons@calculate'])
@option('--prefix', help='Prefix for BB file')
def main(prefix):
    """Calculate Phonon Building Block for QEH model.

    """

    import numpy as np
    from ase.io import read, jsonio
    from gpaw.mpi import world
    from asr.utils import read_json, get_dimensionality
    from ase.phonons import Phonons

    dct = read_json('results-asr.phonons@calculate.json')
    atoms = read('structure.json')
    n = dct['__params__']['n']
    nd = get_dimensionality()
    assert nd == 2, 'Phonon BB can only be calculated for 2D mat'

    supercell = (n, n, 1)
    atoms = read('gs.gpw')
    p = Phonons(atoms=atoms, supercell=supercell)
    p.read()

    C_NN = p.C_N.sum(axis=0)  # For q=0 the Fourier transform is just the sum

    dct = jsonio.decode(read_json('borncharges-0.01.json'))
    Z_avv = dct['Z_avv']
    phbb = phononbuildingblock(atoms=atoms, Z_avv=Z_avv,
                               force_constant_matrix=C_NN)

    formula = atoms.get_chemical_formula('metal')
    if not prefix:
        prefix = ''
    if world.rank == 0:
        np.savez_compressed(f'{prefix}{formula}-phonons.npz', **phbb)


if __name__ == '__main__':
    main.cli()
