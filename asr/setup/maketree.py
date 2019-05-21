from asr.utils import argument, option
import click


@click.command()
@argument('database', nargs=1)
@option('-s', '--selection', help='Selection')
def main(database, selection):
    """Set up folders with atomic structures based on ase-database"""
    from ase.db import connect
    import spglib
    import numpy as np
    
    # from ase import Atoms

    # a = 1.2
    # atoms = Atoms('SMoS', cell=[a, a, 10, 90, 90, 120], pbc=1,
    #               scaled_positions=[[1 / 3, 2 / 3, 0.1],
    #                                 [0, 0, 0],
    #                                 [-1 / 3, -2 / 3, -0.1]])
    # cell = (atoms.cell.array,
    #         atoms.get_scaled_positions(),
    #         atoms.numbers)
    # cell = spglib.standardize_cell(cell)
    # st = atoms.symbols.formula.stoichiometry()[0]
    # dataset = spglib.get_symmetry_dataset(cell)
    # print(dataset)
    # exit()

    if not selection:
        selection = ''
    db = connect(database)
    rows = db.select(selection)

    for row in rows:
        atoms = row.toatoms()
        formula = atoms.symbols.formula
        print(formula)
        for i in [0, 1, 2]:
            spos_ac = atoms.get_scaled_positions()
            spos_ac -= spos_ac[i]
            # spos_ac = np.mod(spos_ac, 1)
            # spos_ac -= spos_ac[0] + [0, 0, 0.5]
            cell = (atoms.cell.array,
                    spos_ac,
                    atoms.numbers)
            print(np.round(cell[1], 2))
            cell = spglib.standardize_cell(cell,
                                           symprec=1e-3)
            print(np.round(cell[1], 2))
            st = atoms.symbols.formula.stoichiometry()[0]
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
            sg = dataset['number']
            wyck = dataset['wyckoffs']
            w = '-'.join(wyck)
            folder = f"{st}/{sg}-{w}/{formula}-{sg}-{w}/"
            print(folder)
        

if __name__ == '__main__':
    main()
