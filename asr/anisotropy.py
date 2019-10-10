import numpy as np
import os.path as op
import gpaw.mpi as mpi
from gpaw.spinorbit import get_anisotropy
from ase.parallel import parprint
from gpaw import GPAW
from asr.utils import command, option

@command('asr.anisotropy')
def main():
    """Calculate the magnetic anisotropy from densk.gpw."""

    if not op.isfile('gs.gpw'):
        return

    width = 0.001
    nbands = None
    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    if mpi.world.rank in ranks:
        calc = GPAW('gs.gpw', communicator=comm, txt=None)
        atoms = calc.atoms
        if calc.wfs.nspins == 1:
            parprint("The ground state is nonmagnetic. "
                     "Skipping anisotropy calculation")
            results = {}
        else:
            E_x = get_anisotropy(calc, theta=np.pi / 2, nbands=nbands, width=width)
            calc = GPAW('gs.gpw', communicator=comm, txt=None)
            E_z = get_anisotropy(calc, theta=0.0, nbands=nbands, width=width)
            calc = GPAW('gs.gpw', communicator=comm, txt=None)
            E_y = get_anisotropy(
                calc, theta=np.pi / 2, phi=np.pi / 2, nbands=nbands, width=width)

            dE_zx = E_z - E_x
            dE_zy = E_z - E_y

            results = {'dE_zx': dE_zx, 'dE_zy': dE_zy,
                       '__key_descriptions__':
                       {'dE_zx': 'KVP: Magnetic anisotropy energy (zx-component) [meV/formula unit]',
                       'dE_zy': 'KVP: Magnetic anisotropy energy (zy-component) [meV/formula unit]'}}

        return results


def webpanel(row, key_descriptions):

    from asr.utils.custom import table
    if row.magstate != 'NM':
        magtable = table(row, 'Property',
                         ['magstate', 'magmom',
                          'maganis_zx', 'maganis_zy', 'dE_NM'])
        panel = ('Magnetic properties', [[magtable], []])
    else:
        panel = []
    things = ()

    return panel, things

#group = 'postprocessing'
dependencies = ['asr.structureinfo', 'asr.gs']

if __name__ == '__main__':
    main.cli()