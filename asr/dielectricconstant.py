from asr.core import command, option
from click import Choice


@command('asr.dielectricconstant')
@option(
    '--gs', help='Ground state on which response is based')
@option('--kptdensity', help='K-point density')
@option('--ecut', help='Plane wave cutoff')
@option('--xc', help='XC interaction',
        type=Choice(['RPA', 'ALDA']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
# @option('--low_freq/--high_freq', default=True,
#         help='Specify which frequency limit to apply')
def main(gs='gs.gpw', kptdensity=20.0, ecut=50.0, xc='RPA', bandfactor=5):
    """Calculate static dielectric constant.

    This recipe calculates to low-frequency (relaxed-ion) dielectric constant
    (frequency=0), e.g. used for defect calculations when the defect geometry
    is allowed to relax (H.-P. Komsa, T. T. Rantala and A. Pasquarello Phys.
    Rev. B 86, 045112 (2012)).
    """
    from ase.io import read
    from asr.utils import write_json
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.df import DielectricFunction
    from gpaw.occupations import FermiDirac
    from pathlib import Path
    import numpy as np

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


def collect_data(atoms):
    from pathlib import Path
    from ase.io import jsonio
    if not Path('dielectricconstant.json').is_file():
        return {}, {}, {}

    kvp = {}
    key_descriptions = {}
    dct = jsonio.decode(Path('dielectricconstant.json').read_text())

    # Update key-value-pairs
    kvp['epsilon_LF_x'] = dct['local_field'][0]
    kvp['epsilon_LF_y'] = dct['local_field'][1]
    kvp['epsilon_LF_z'] = dct['local_field'][2]
    kvp['epsilon_NLF_x'] = dct['no_local_field'][0]
    kvp['epsilon_NLF_y'] = dct['no_local_field'][1]
    kvp['epsilon_NLF_z'] = dct['no_local_field'][2]

    # Update key_descriptions
    kd = {
        'epsilon_LF': ('Static dielectric constant (with local field '
                       f'correction [x, y, z]-direction'),
        'epsilon_NLF': ('Static dielectric constant (without local field '
                        f'correction [x, y, z]-direction')
    }
    key_descriptions.update(kd)

    # Save data
    return kvp, key_descriptions, dct


def webpanel():
    return None


def postprocessing():
    """Extract data from dielectricconstant.json.

    This will be called after main by default."""
    from asr.utils import read_json
    results = read_json('dielectricconstant.json')
    results['__key_descriptions__'] = {
        'local_field': 'Static dielectric constant with local field '
        f'correction [x, y, z]-direction',
        'no_local_field': 'Static dielectric constant without local '
        f'field correction [x, y, z]-direction'}

    return results


group = 'property'
creates = ['dielectricconstant.json']
# dependencies = ['asr.structureinfo', 'asr.gs']

if __name__ == '__main__':
    main()
