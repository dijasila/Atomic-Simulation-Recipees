from asr.utils import update_defaults, command, option
import click

@command()
@option('--gs', default='gs.gpw', help='Ground state on which response is based')
@option('--kptpath', default=None, help='k-point path')
@option('--width', default=0.001, help='Fermi Dirac width')
@option('--nbands', default=None, type=int, help='Number of bands')

def main(gs, kptpath, width, nbands):
    """Calculate the magnetic anisotropy from densk.gpw."""


    import os.path as op
    from gpaw import GPAW
    import gpaw.mpi as mpi
    from gpaw.spinorbit import get_anisotropy
    from ase.parallel import paropen, parprint
    import numpy as  np

    from pathlib import Path
    import json
    from ase.io import jsonio

    if not op.isfile(gs): ###### densk.gpw
        return

    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    if mpi.world.rank in ranks:
        calc = GPAW(gs, communicator=comm, txt=None)
        if calc.wfs.nspins == 1:
            parprint("The ground state is nonmagnetic. "
                     "Skipping anisotropy calculation")
            return
        E_x = get_anisotropy(calc, theta=np.pi / 2, nbands=nbands, width=width)
        calc = GPAW(gs, communicator=comm, txt=None)
        E_z = get_anisotropy(calc, theta=0.0, nbands=nbands, width=width)
        calc = GPAW(gs, communicator=comm, txt=None)
        E_y = get_anisotropy(
            calc, theta=np.pi / 2, phi=np.pi / 2, nbands=nbands, width=width)

        dE_zx = E_z - E_x
        dE_zy = E_z - E_y

        data = {
                'dE_zx': dE_zx,
                'dE_zy': dE_zy,
        }
        
        filename = 'anisotropy_xy.json'

        Path(filename).write_text(json.dumps(data, cls=jsonio.MyEncoder))



def collect_data(atoms):
    """Collect data to ASE database"""

    from pathlib import Path
    from ase.io import jsonio
    if not Path('anisotropy_xy.json').is_file():
        return {}, {}, {}

    kvp = {}
    data = {}
    key_descriptions = {}
    dct = jsonio.decode(Path('anisotropy_xy.json').read_text())

    # Update key-value-pairs
    kvp['dE_zx'] = dct['dE_zx']
    kvp['dE_zy'] = dct['dE_zy']

    # Update key descriptions
    kd = {
        'dE_zx': ('Magnetic anisotropy energy (xz-component)',
                  'Magnetic anisotropy energy (xz-component)', 'meV / formula unit'),
        'dE_zy': ('Magnetic anisotropy energy (yz-component)',
                  'Magnetic anisotropy energy (yz-component)', 'meV / formula unit')
        }

    key_descriptions.update(kd)

    # Save data
    data['anisotropy'] = dct

    return kvp, key_descriptions, data


def webpanel(row, key_descriptions):
    from asr.custom import table
    if row.magstate != 'NM':
        magtable = table('Property',
                         ['magstate', 'magmom',
                          'maganis_zx', 'maganis_zy', 'dE_NM'])
        panel = ('Magnetic properties', [[magtable], []])
    else:
        panel = []

    things = ()
    return panel, things


group = 'Property'
dependencies = ['asr.gs'] #densk

if __name__ == '__main__':
    main(standalone_mode=False)
