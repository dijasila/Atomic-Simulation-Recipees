from asr.core import command


def webpanel(row, key_descriptions):
    from asr.browser import table
    magtable = table(row, 'Property',
                     ['magstate', 'magmom',
                      'maganis_zx', 'maganis_zy'])
    panel = {'title': 'Magnetic properties',
             'columns': [[magtable], []]}
    return [panel]


params = '''asr.gs@calculate:calculator {'mode':'lcao','kpts':(2,2,2),...}'''
tests = [{'cli': ['ase build -x hcp Co structure.json',
                  f'asr run "setup.params {params}"',
                  'asr run asr.anisotropy',
                  'asr run database.fromtree',
                  'asr run "browser --only-figures"']}]


@command('asr.anisotropy',
         tests=tests,
         requires=['gs.gpw'],
         webpanel=webpanel,
         dependencies=['asr.gs@calculate'])
def main():
    """Calculate the magnetic anisotropy."""
    import numpy as np
    from gpaw.spinorbit import get_anisotropy
    from gpaw import GPAW
    from gpaw.mpi import serial_comm

    width = 0.001
    nbands = None
    calc = GPAW('gs.gpw', communicator=serial_comm, txt=None)
    E_x = get_anisotropy(calc, theta=np.pi / 2, nbands=nbands,
                         width=width)
    calc = GPAW('gs.gpw', communicator=serial_comm, txt=None)
    E_z = get_anisotropy(calc, theta=0.0, nbands=nbands, width=width)
    calc = GPAW('gs.gpw', communicator=serial_comm, txt=None)
    E_y = get_anisotropy(calc, theta=np.pi / 2, phi=np.pi / 2,
                         nbands=nbands, width=width)

    dE_zx = E_z - E_x
    dE_zy = E_z - E_y

    results = {'dE_zx': dE_zx, 'dE_zy': dE_zy,
               '__key_descriptions__':
               {'dE_zx': ('KVP: Magnetic anisotropy energy '
                          '(zx-component) [meV/formula unit]'),
                'dE_zy': ('KVP: Magnetic anisotropy energy '
                          '(zy-component) [meV/formula unit]')}}

    return results


if __name__ == '__main__':
    main.cli()
