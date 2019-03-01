from ase.io import read
from ase.io.ulm import open as ulmopen
from gpaw import GPAW, PW, FermiDirac


def write_gpw_file():
    params = dict(
        mode=PW(800),
        xc='PBE',
        basis='dzp',
        kpts={'density': 6.0, 'gamma': True},
        occupations=FermiDirac(width=0.05))

    name = 'start.traj'
    u = ulmopen(name)
    params.update(u[-1].calculator.get('parameters', {}))
    u.close()
    slab = read(name)
    slab.calc = GPAW(txt='gs.txt', **params)
    slab.get_forces()
    slab.get_stress()
    slab.calc.write('gs.gpw')

    
def get_parser():
    import argparse
    desc = 'Calculate ground state density and save to gs.gpw'
    parser = argparse.ArgumnentParser(description=desc)

    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args()
    write_gpw_file(**args)


if __name__ == '__main__':
    main()
