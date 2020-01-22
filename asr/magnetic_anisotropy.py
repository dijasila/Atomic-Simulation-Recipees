from asr.core import command, read_json


def get_spin_axis():
    anis = read_json('results-asr.magnetic_anisotropy.json')
    return anis['theta'], anis['phi']


def get_spin_index():
    anis = read_json('results-asr.magnetic_anisotropy.json')
    axis = anis['spin_axis']
    if axis == 'z':
        index = 2
    elif axis == 'y':
        index = 1
    else:
        index = 0
    return index


def spin_axis(theta, phi):
    import numpy as np
    if theta == 0:
        return 'z'
    elif np.allclose(phi, np.pi / 2):
        return 'y'
    else:
        return 'x'


def webpanel(row, key_descriptions):
    from asr.database.browser import table
    magtable = table(row, 'Property',
                     ['magstate', 'magmom',
                      'dE_zx', 'dE_zy'], kd=key_descriptions)
    panel = {'title': 'Basic magnetic properties (PBE)',
             'columns': [[magtable], []],
             'sort': 11}
    return [panel]


params = '''asr.gs@calculate:calculator +{'mode':'lcao','kpts':(2,2,2)}'''
tests = [{'cli': ['ase build -x hcp Co structure.json',
                  f'asr run "setup.params {params}"',
                  'asr run asr.magnetic_anisotropy',
                  'asr run database.fromtree',
                  'asr run "database.browser --only-figures"']}]


@command('asr.magnetic_anisotropy',
         tests=tests,
         requires=['gs.gpw', 'results-asr.structureinfo.json'],
         webpanel=webpanel,
         dependencies=['asr.gs@calculate', 'asr.structureinfo'])
def main():
    """Calculate the magnetic anisotropy.
    Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Returns:
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
    """
    import numpy as np
    from asr.core import file_barrier, read_json
    from gpaw.mpi import world
    from gpaw.spinorbit import get_anisotropy
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from gpaw.utilities.ibz2bz import ibz2bz
    from pathlib import Path

    structureinfo = read_json('results-asr.structureinfo.json')
    magstate = structureinfo['magstate']
    # Figure out if material is magnetic
    results = {'__key_descriptions__':
               {'spin_axis': 'KVP: Suggested spin direction for SOC',
                'E_x': 'KVP: SOC total energy difference in x-direction',
                'E_y': 'KVP: SOC total energy difference in y-direction',
                'E_z': 'KVP: SOC total energy difference in z-direction',
                'theta': 'Spin direction, theta, polar coordinates [radians]',
                'phi': 'Spin direction, phi, polar coordinates [radians]',
                'dE_zx': ('KVP: Magnetic anisotropy energy '
                          '(zx-component) [meV/formula unit]'),
                'dE_zy': ('KVP: Magnetic anisotropy energy '
                          '(zy-component) [meV/formula unit]')}}

    if magstate == 'NM':
        results['E_x'] = 0
        results['E_y'] = 0
        results['E_z'] = 0
        results['dE_zx'] = 0
        results['dE_zy'] = 0
        results['theta'] = 0
        results['phi'] = 0
        results['spin_axis'] = 'z'
        return results

    with file_barrier(['gs_nosym.gpw']):
        ibz2bz('gs.gpw', 'gs_nosym.gpw')
    width = 0.001
    nbands = None
    calc = GPAW('gs_nosym.gpw', communicator=serial_comm, txt=None)
    E_x = get_anisotropy(calc, theta=np.pi / 2, nbands=nbands,
                         width=width)
    calc = GPAW('gs_nosym.gpw', communicator=serial_comm, txt=None)
    E_z = get_anisotropy(calc, theta=0.0, nbands=nbands, width=width)
    calc = GPAW('gs_nosym.gpw', communicator=serial_comm, txt=None)
    E_y = get_anisotropy(calc, theta=np.pi / 2, phi=np.pi / 2,
                         nbands=nbands, width=width)

    dE_zx = E_z - E_x
    dE_zy = E_z - E_y

    DE = max(dE_zx, dE_zy)
    theta = 0
    phi = 0
    if DE > 0:
        theta = np.pi / 2
        if dE_zy > dE_zx:
            phi = np.pi / 2

    axis = spin_axis(theta, phi)

    results.update({'spin_axis': axis,
                    'theta': theta,
                    'phi': phi,
                    'E_x': E_x,
                    'E_y': E_y,
                    'E_z': E_z,
                    'dE_zx': dE_zx,
                    'dE_zy': dE_zy})
    world.barrier()
    if world.rank == 0:
        Path('gs_nosym.gpw').unlink()
    return results


if __name__ == '__main__':
    main.cli()
