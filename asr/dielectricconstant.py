from asr.utils import command, option
from click import Choice


@command('asr.dielectricconstant')
@option(
    '--gs', default='gs.gpw', help='Ground state on which response is based')
@option('--kptdensity', default=20.0, help='K-point density')
@option('--ecut', default=50.0, help='Plane wave cutoff')
@option('--xc', default='RPA', help='XC interaction',
        type=Choice(['RPA', 'ALDA']))
@option('--bandfactor', default=5, type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
@option('--low_freq/--high_freq', default=True, 
        help='Specify which frequency limit to apply')
def main(gs, kptdensity, ecut, xc, bandfactor, low_freq):
    """Calculate static dielectric constant."""
    import json
    from ase.io import jsonio, read
    from asr.utils import write_json
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.df import DielectricFunction
    from gpaw.occupations import FermiDirac
    from pathlib import Path
    import numpy as np
    from math import pi

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()

    dfkwargs = {
        'eta': 0.05,
        'domega0': 0.005,
        'ecut': ecut,
        'name': 'chi',
        'intraband': False
    }

    ND = np.sum(pbc)
    if ND == 3 or ND == 1:
        kpts = {'density': kptdensity, 'gamma': False, 'even': True}
    elif ND == 2:

        def get_kpts_size(atoms, density):
            """trying to get a reasonable monkhorst size which hits high
            symmetry points
            """
            from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
            size, offset = k2so(atoms=atoms, density=density)
            size[2] = 1
            for i in range(2):
                if size[i] % 6 != 0:
                    size[i] = 6 * (size[i] // 6 + 1)
            kpts = {'size': size, 'gamma': True}
            return kpts

        kpts = get_kpts_size(atoms=atoms, density=kptdensity)
        volume = atoms.get_volume()
        if volume < 120:
            nblocks = world.size // 4
        else:
            nblocks = world.size // 2
        dfkwargs.update({
            'nblocks': nblocks,
            'pbc': pbc,
            'integrationmode': 'tetrahedron integration',
            'truncation': '2D'
        })

    else:
        raise NotImplementedError(
            'Polarizability not implemented for 1D and 2D structures')

    if not Path('es.gpw').is_file():
        calc_old = GPAW(gs, txt=None)
        nval = calc_old.wfs.nvalence

        calc = GPAW(
            gs,
            txt='es.txt',
            fixdensity=True,
            nbands=(bandfactor + 1) * nval,
            convergence={'bands': bandfactor * nval},
            occupations=FermiDirac(width=1e-4),
            kpts=kpts)
        calc.get_potential_energy()
        calc.write('es.gpw', mode='all')

    df = DielectricFunction('es.gpw', **dfkwargs)

    epsNLF_x, epsLF_x = df.get_macroscopic_dielectric_constant(direction='x')
    epsNLF_y, epsLF_y = df.get_macroscopic_dielectric_constant(direction='y')
    epsNLF_z, epsLF_z = df.get_macroscopic_dielectric_constant(direction='z')

    epsilon = {}
    filename_eps = 'dielectricconstant.json'
    epsilon['local_field'] = [epsLF_x, epsLF_y, epsLF_z]
    epsilon['no_local_field'] = [epsNLF_x, epsNLF_y, epsNLF_z]
    write_json(filename_eps, epsilon)


def postprocessing():
    return None


def collect_data(atoms):
    from pathlib import Path
    from ase.io import jsonio
    from math import pi
    if not Path('polarizability.json').is_file():
        return {}, {}, {}

    kvp = {}
    data = {}
    key_descriptions = {}
    dct = jsonio.decode(Path('polarizability.json').read_text())
    
    # Update key-value-pairs
    kvp['epsilonx'] = (1 + 4 * pi) * dct['alphax_w'][0].real
    kvp['epsilony'] = (1 + 4 * pi) * dct['alphay_w'][0].real
    kvp['epsilonz'] = (1 + 4 * pi) * dct['alphaz_w'][0].real

    # Update key_descriptions
    kd = {
        'epsilonx': ('Static dielectric constant (x-direction)',
                     'Static dielectric constant (x-direction)', 'Ang'),
        'epsilony': ('Static dielectric constant (y-direction)',
                     'Static dielectric constant (y-direction)', 'Ang'),
        'epsilonz': ('Static dielectric constant (z-direction)',
                     'Static dielectric constant (z-direction)', 'Ang')
    }
    key_descriptions.update(kd)

    # Save data
    data['absorptionspectrum'] = dct
    return kvp, key_descriptions, data


def webpanel():
    return None


group = 'property'
creates = ['dielectricconstant.json']
#dependencies = ['asr.structureinfo', 'asr.gs']

if __name__ == '__main__':
    main()
