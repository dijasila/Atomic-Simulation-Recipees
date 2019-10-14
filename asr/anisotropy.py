from asr.core import command


def webpanel(row, key_descriptions):
    from asr.browser import table
    magtable = table(row, 'Property',
                     ['magstate', 'magmom',
                      'maganis_zx', 'maganis_zy', 'dE_NM'])
    panel = {'title': 'Magnetic properties',
             'columns': [[magtable], []]}
    return [panel]


@command('asr.anisotropy',
         requires=['gs.gpw'],
         webpanel=webpanel,
         dependencies=['asr.structureinfo', 'asr.gs'])
def main():
    """Calculate the magnetic anisotropy from densk.gpw."""
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
