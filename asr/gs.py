from asr.utils import command, option
from pathlib import Path


calctests = [{'description': 'Test ground state of Si.',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params asr.gs@calculate:ecut 300 '
                      'asr.gs@calculate:kptdensity 2"',
                      'asr run gs@calculate',
                      'asr run database.fromtree',
                      'asr run "browser --only-figures"']}]


@command(module='asr.gs',
         creates=['gs.gpw'],
         tests=calctests,
         requires=['structure.json'],
         resources='8:10h',
         restart=1)
@option('--ecut', type=float, help='Plane-wave cutoff')
@option('-k', '--kptdensity', type=float, help='K-point density')
@option('--xc', type=str, help='XC-functional')
@option('--width', help='Fermi-Dirac smearing temperature')
@option('-r', '--readoutcharge', type=bool,
        help='Read out chargestate from params.json')
def calculate(ecut=800, xc='PBE',
              kptdensity=12.0, width=0.05, readoutcharge=False): 
    """Calculate ground state file.
    This recipe saves the ground state to a file gs.gpw based on the structure
    in 'structure.json'. This can then be processed by asr.gs@postprocessing
    for storing any derived quantities. See asr.gs@postprocessing for more
    information."""
    import numpy as np
    from ase.io import read
    from asr.calculators import get_calculator
    from asr.utils import read_json

    atoms = read('structure.json')

    # Read out chargestate from params.json if specified as option
    if readoutcharge:
        setup_params = read_json('params.json')
        chargestate = setup_params.get('charge')
        print('INFO: chargestate {}'.format(chargestate))
    else:
        chargestate = 0

    params = dict(
        mode={'name': 'pw', 'ecut': ecut},
        xc=xc,
        basis='dzp',
        kpts={
            'density': kptdensity,
            'gamma': True
        },
        occupations={'name': 'fermi-dirac', 'width': width},
        convergence={'bands': -3},
        txt='gs.txt',
        charge=chargestate)

    nd = np.sum(atoms.pbc)
    if nd == 2:
        assert not atoms.pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        params['poissonsolver'] = {'dipolelayer': 'xy'}
    elif nd == 1:
        assert not atoms.pbc[0] and not atoms.pbc[1]

    calc = get_calculator()(**params)

    atoms.calc = calc
    atoms.get_forces()
    atoms.get_stress()
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw')


tests = [{'description': 'Test ground state of Si.',
          'cli': ['asr run "setup.materials -s Si2"',
                  'ase convert materials.json structure.json',
                  'asr run "setup.params asr.gs@calculate:ecut 300 '
                  'asr.gs@calculate:kptdensity 2"',
                  'asr run gs',
                  'asr run database.fromtree',
                  'asr run "browser --only-figures"']}]


@command(module='asr.gs',
         requires=['gs.gpw'],
         tests=tests,
         dependencies=['asr.gs@calculate'])
def main():
    """Extract derived quantities from groundstate in gs.gpw."""
    import numpy as np
    from asr.calculators import get_calculator

    # Just some quality control before we start
    calc = get_calculator()('gs.gpw', txt=None)
    pbc = calc.atoms.pbc
    ndim = np.sum(pbc)

    if ndim == 2:
        assert not pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        # For 2D materials we check that the calculater used a dipole
        # correction if the material has an out-of-plane dipole

        # Small hack
        atoms = calc.atoms
        atoms.calc = calc
        evacdiffmin = 10e-3
        if evacdiff(calc.atoms) > evacdiffmin:
            assert calc.todict().get('poissonsolver', {}) == \
                {'dipolelayer': 'xy'}, \
                ('The ground state has a finite dipole moment along aperiodic '
                 'axis but calculation was without dipole correction.')
    elif nd == 1:
        assert not atoms.pbc[0] and not atoms.pbc[1]

    # Now that some checks are done, we can extract information
    forces = calc.get_property('forces', allow_calculation=False)
    stresses = calc.get_property('stress', allow_calculation=False)
    etot = calc.get_potential_energy()

    results = {'forces': forces,
               'stresses': stresses,
               'etot': etot,
               'gs.gpw': str(Path('gs.gpw').absolute()),
               '__key_descriptions__':
               {'forces': 'Forces on atoms [eV/Angstrom]',
                'stresses': 'Stress on unit cell [eV/Angstrom^dim]',
                'etot': 'KVP: Total energy (En.) [eV]',
                'evac': 'KVP: Vacuum level (Vacuum level) [eV]'}}

    analysegs('gs.gpw', results)

    fingerprint = {}
    for setup in calc.setups:
        fingerprint[setup.symbol] = setup.fingerprint
    results['__setup_fingerprints__'] = fingerprint

    return results


def analysegs(gpw, results):
    """Analyse the computed ground state according to the class of materials,
    it belongs to."""
    # What about metals? XXX - should dos_at_ef be here?
    # There should be a metals check somewhere? XXX
    from pathlib import Path
    import numpy as np
    from gpaw import restart

    if not Path(gpw).is_file():
        raise ValueError('Groundstate file not present')
    atoms, calc = restart(gpw, txt=None)

    results['gaps_nosoc'] = gaps(calc, gpw, soc=False)
    results['gaps_soc'] = gaps(calc, gpw, soc=True)

    # Vacuum level is calculated for c2db backwards compability
    if int(np.sum(atoms.get_pbc())) == 2:
        vac = vacuumlevels(atoms, calc)
        results['vacuumlevels'] = vac
        results['evac'] = vac['evacmean']
        results['evacdiff'] = vac['evacdiff']


def gaps(calc, gpw, soc=True):
    """Could use some documentation!!! XXX
    Who is in charge of this thing??
    """
    # ##TODO min kpt dens? XXX
    # inputs: gpw groundstate file, soc?, direct gap? XXX
    from functools import partial
    from asr.utils.gpw2eigs import gpw2eigs

    ibzkpts = calc.get_ibz_k_points()

    (evbm_ecbm_gap,
     skn_vbm, skn_cbm) = get_gap_info(soc=soc, direct=False,
                                      calc=calc, gpw=gpw)
    (evbm_ecbm_direct_gap,
     direct_skn_vbm, direct_skn_cbm) = get_gap_info(soc=soc, direct=True,
                                                    calc=calc, gpw=gpw)

    k_vbm, k_cbm = skn_vbm[1], skn_cbm[1]
    direct_k_vbm, direct_k_cbm = direct_skn_vbm[1], direct_skn_cbm[1]

    get_kc = partial(get_1bz_k, ibzkpts, calc)

    k_vbm_c = get_kc(k_vbm)
    k_cbm_c = get_kc(k_cbm)
    direct_k_vbm_c = get_kc(direct_k_vbm)
    direct_k_cbm_c = get_kc(direct_k_cbm)

    if soc:
        _, efermi = gpw2eigs(gpw, soc=True,
                             optimal_spin_direction=True)
    else:
        efermi = calc.get_fermi_level()

    subresults = {'gap': evbm_ecbm_gap[2],
                  'vbm': evbm_ecbm_gap[0],
                  'cbm': evbm_ecbm_gap[1],
                  'gap_dir': evbm_ecbm_direct_gap[2],
                  'vbm_dir': evbm_ecbm_direct_gap[0],
                  'cbm_dir': evbm_ecbm_direct_gap[1],
                  'k1_c': k_vbm_c,
                  'k2_c': k_cbm_c,
                  'k1_dir_c': direct_k_vbm_c,
                  'k2_dir_c': direct_k_cbm_c,
                  'skn1': skn_vbm,
                  'skn2': skn_cbm,
                  'skn1_dir': direct_skn_vbm,
                  'skn2_dir': direct_skn_cbm,
                  'efermi': efermi}

    return subresults


def get_1bz_k(ibzkpts, calc, k_index):
    from gpaw.kpt_descriptor import to1bz
    k_c = ibzkpts[k_index] if k_index is not None else None
    if k_c is not None:
        k_c = to1bz(k_c[None], calc.wfs.gd.cell_cv)[0]
    return k_c


def get_gap_info(soc, direct, calc, gpw):
    from ase.dft.bandgap import bandgap
    from asr.utils.gpw2eigs import gpw2eigs
    # e1 is VBM, e2 is CBM
    if soc:
        e_km, efermi = gpw2eigs(gpw, soc=True, optimal_spin_direction=True)
        # km1 is VBM index tuple: (s, k, n), km2 is CBM index tuple: (s, k, n)
        gap, km1, km2 = bandgap(eigenvalues=e_km, efermi=efermi, direct=direct,
                                kpts=calc.get_ibz_k_points(), output=None)
        if km1[0] is not None:
            e1 = e_km[km1]
            e2 = e_km[km2]
        else:
            e1, e2 = None, None
        x = (e1, e2, gap), (0,) + tuple(km1), (0,) + tuple(km2)
    else:
        g, skn1, skn2 = bandgap(calc, direct=direct, output=None)
        if skn1[1] is not None:
            e1 = calc.get_eigenvalues(spin=skn1[0], kpt=skn1[1])[skn1[2]]
            e2 = calc.get_eigenvalues(spin=skn2[0], kpt=skn2[1])[skn2[2]]
        else:
            e1, e2 = None, None
        x = (e1, e2, g), skn1, skn2
    return x


def vacuumlevels(atoms, calc, n=8):
    """Get the vacuumlevels on both sides of a 2D material. Will
    do a dipole corrected dft calculation, if needed (Janus structures).
    Assumes the 2D material periodic directions are x and y.
    Assumes that the 2D material is centered in the z-direction of
    the unit cell.

    Dipole corrected dft calculation -> dipcorrgs.gpw

    Parameters
    ----------
    gpw: str
       name of gpw file to base the dipole corrected calc on
    evacdiffmin: float
        thresshold in eV for doing a dipole moment corrected
        dft calculations if the predicted evac difference is less
        than this value don't do it
    n: int
        number of gridpoints away from the edge to evaluate the vac levels
    """
    import numpy as np
 
    # Record electrostatic potential as a function of z
    v_z = calc.get_electrostatic_potential().mean(0).mean(0)
    z_z = np.linspace(0, atoms.cell[2, 2], len(v_z), endpoint=False)

    # Store data
    subresults = {'z_z': z_z, 'v_z': v_z,
                  'evacdiff': evacdiff(atoms),
                  'dipz': atoms.get_dipole_moment()[2],
                  # Extract vaccuum energy on both sides of the slab
                  'evac1': v_z[n],
                  'evac2': v_z[-n],
                  'evacmean': (v_z[n] + v_z[-n]) / 2,
                  # Ef might have changed in dipole corrected gs
                  'efermi_nosoc': calc.get_fermi_level()}

    return subresults


def evacdiff(atoms):
    """Calculate vacuum energy level difference from the dipole moment of
    a slab assumed to be in the xy plane

    Returns
    -------
    out: float
        vacuum level difference in eV
    """
    import numpy as np
    from ase.units import Bohr, Hartree

    A = np.linalg.det(atoms.cell[:2, :2] / Bohr)
    dipz = atoms.get_dipole_moment()[2] / Bohr
    evacsplit = 4 * np.pi * dipz / A * Hartree

    return evacsplit


def get_evac():
    """Get mean vacuum energy, if it has been calculated"""
    from pathlib import Path
    from asr.utils import read_json

    evac = None
    if Path('results-asr.gs.json').is_file():
        results = read_json('results-asr.gs.json')
        if 'vacuumlevels' in results.keys():
            evac = results['vacuumlevels']['evacmean']

    return evac


def webpanel(row, key_descriptions):
    from asr.utils.custom import table

    if row.get('evacdiff', 0) < 0.02:
        t = table(row, 'Postprocessing',
                  ['gap', 'vbm', 'cbm',
                   'gap_dir', 'vbm_dir', 'cbm_dir', 'efermi'],
                  key_descriptions)
    else:
        t = table(row, 'Postprocessing',
                  ['gap', 'vbm', 'cbm',
                   'gap_dir', 'vbm_dir', 'cbm_dir', 'efermi',
                   'dipz', 'evacdiff'],
                  key_descriptions)

    panel = {'title': 'PBE ground state',
             'columns': [[t]]}

    return [panel]


if __name__ == '__main__':
    main.cli()
