import click


@click.command()
@click.argument('filename')
@click.argument('destination')
@click.argument('vacuum', default=7.5)
def main(filename, destination, vacuum=7.5):
    """Prepare structure for recipes"""
    from ase.io import read, write
    from ase.build import niggli_reduce
    import numpy as np
    atoms = read(filename)
    pbc = atoms.get_pbc()
    if not np.all(pbc):
        axis = np.argwhere(~pbc)[0]
        atoms.center(vacuum=vacuum,
                     axis=axis)
    atoms.pbc = (1, 1, 1)
    niggli_reduce(atoms)
    atoms.pbc = pbc
    write(destination, atoms)


if __name__ == '__main__':
    main()
