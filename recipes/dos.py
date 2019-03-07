def calculate(name='dos.gpw'):
    """
    Calculate DOS
    """
    from gpaw import GPAW
    calc = GPAW('gs.gpw', txt='dos.txt',
                kpts={'density': 12.0},
                nbands='300%',
                convergence={'bands': -10})
    calc.get_potential_energy()
    calc.write(name)
    del calc
    
    calc = GPAW(name, txt=None)
    from ase.dft.dos import DOS
    dos = DOS(calc, width=0.0, window=(-5, 5), npts=1000)
    dosspin0_e = dos.get_dos(spin=0)
    dosspin1_e = dos.get_dos(spin=1)
    energies_e = dos.get_energies()
    
    data = {'dosspinup_e': dosspin0_e,
            'dosspindown_e': dosspin1_e,
            'energies_e': energies_e}
    import json
    filename = 'dos.json'
    
    from ase.parallel import paropen
    with paropen(filename, 'w') as fd:
        json.dump(data, fd)


def get_parser():
    import argparse
    desc = 'Calculate density of states'
    parser = argparse.ArgumentParser(description=desc)
    help = 'Name of calculator to calculate DOS from'
    parser.add_argument('-n', '--name', type=str,
                        default='dos.gpw',
                        help=help)
    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = vars(parser.parse_args())
    calculate(**args)


if __name__ == '__main__':
    main()
