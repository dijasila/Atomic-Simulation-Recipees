from asr.utils import click, update_defaults


@click.command()
@update_defaults('asr.gs')
@click.option('-a', '--atomfile', type=str, help='Atomic structure',
              default='start.traj')
@click.option('--gpwfilename', type=str, help='filename.gpw',
              default='gs.gpw')
@click.option('--ecut', type=float, help='Plane-wave cutoff',
              default=800)
@click.option('-k', '--kptdensity', type=float, help='K-point density',
              default=6.0)
@click.option('--xc', type=str, help='XC-functional',
              default='PBE')
def main(atomfile, gpwfilename, ecut, xc, kptdensity):
    """Calculate ground state density"""
    from ase.io import read
    from gpaw import GPAW, PW, FermiDirac
    params = dict(
        mode=PW(ecut),
        xc=xc,
        basis='dzp',
        kpts={'density': kptdensity, 'gamma': True},
        occupations=FermiDirac(width=0.05),
        txt='gs.txt')

    slab = read(atomfile)
    slab.calc = GPAW(**params)
    slab.get_forces()
    slab.get_stress()
    slab.calc.write(gpwfilename)


# The metadata is put it the bottom
group = 'Property'
description = ''
dependencies = []  # What other recipes does this recipe depend on
creates = ['gs.gpw']  # What files are created
resources = '8:1h'  # How many resources are used
diskspace = 0  # How much diskspace is used
restart = 1  # Does it make sense to restart the script?


if __name__ == '__main__':
    main()
