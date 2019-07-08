from asr.utils import command, option


# ---------- Main functionality ---------- #


@command('asr.analysegs')
@option('--gpw', type=str, help='filename.gpw', default='gs.gpw')
def main(gpw):
    from pathlib import Path
    import numpy as np
    from gpaw import restart

    if not Path(gpw).is_file():
        raise ValueError('Groundstate file not present')
    atoms, calc = restart(gpw, txt=None)

    results = {}

    results['gaps_nosoc'] = gaps(calc, gpw, soc=False)
    results['gaps_soc'] = gaps(calc, gpw, soc=True)

    # Vacuum level is calculated for c2db backwards compability
    if int(np.sum(atoms.get_pbc())) == 2:  # dimensionality = 2
        results['vacuumlevels'] = vacuumlevels(atoms, calc, gpw=gpw)

    return results
    

# ---------- Recipe methodology ---------- #


# ----- gaps ----- #


def gaps(calc, gpw, soc=True):
    """Could use some documentation!!! XXX
    Who is in charge of this thing??
    """
    # ##TODO min kpt dens? XXX
    # inputs: gpw groundstate file, soc?, direct gap? XXX
    from functools import partial
    from pathlib import Path
    from ase.parallel import paropen
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


# ----- vacuumlevels ----- #


def vacuumlevels(atoms, calc, gpw='gs.gpw', evacdiffmin=10e-3):
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
    """
    p = calc.todict()
    if p.get('poissonsolver', {}) == {'dipolelayer': 'xy'}\
       or abs(evacdiff(atoms)) < evacdiffmin:
        atoms, calc = dipolecorrectedgs(gpw)

    return calculate_evac(atoms, calc)


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


def dipolecorrectedgs(gpw='gs.gpw'):
    """Do dipole corrected ground state calculation."""
    from gpaw import GPAW

    calc = GPAW(gpw, txt='dipcorrgs.txt')
    calc.set(poissonsolver={'dipolelayer': 'xy'})
    atoms = calc.get_atoms()
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()
    calc.write('dipcorrgs.gpw')

    return atoms, calc


def calculate_evac(atoms, calc, n=8):
    """Calculate the vacuumlevels on both sides of the 2D material.

    Parameters
    ----------
    n: int
        number of gridpoints away from the edge to evaluate the vac levels
    """
    import numpy as np
    from ase.parallel import world

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
                  # Ef might have changed in dipole corrected gs
                  'efermi_nosoc': calc.get_fermi_level()}

    return subresults


# ---------- Database and webpanel ---------- #


# Old format
def collect_data(atoms):
    kvp = {}
    kd = {}
    data = {}

    from asr.utils import read_json
    results = read_json('results_analysegs.json')

    # ----- gaps ----- #
    
    data_to_include = ['gap', 'vbm', 'cbm', 
                       'gap_dir', 'vbm_dir', 'cbm_dir', 'efermi']
    # What about description of non kvp data? XXX
    descs = [('Bandgap', 'Bandgap', 'eV'),
             ('Valence Band Maximum', 'Maximum of valence band', 'eV'),
             ('Conduction Band Minimum', 'Minimum of conduction band', 'eV'),
             ('Direct Bandgap', 'Direct bandgap', 'eV'),
             ('Valence Band Maximum - Direct',
              'Valence Band Maximum - Direct', 'eV'),
             ('Conduction Band Minimum - Direct',
              'Conduction Band Minimum - Direct', 'eV'),
             ('Fermi Level', "Fermi's level", 'eV')]

    for soc in [True, False]:
        socname = 'soc' if soc else 'nosoc'
        keyname = f'gaps_{socname}'
        subresults = results[keyname]

        def namemod(n):
            return n + '_soc' if soc else n

        includes = [namemod(n) for n in data_to_include]

        for k, inc in enumerate(includes):
            val = subresults[data_to_include[k]]
            if val is not None:
                kvp[inc] = val
                kd[inc] = descs[k]

    # ----- vacuumlevels ----- #

    if 'vacuumlevels' in results.keys():
        subresults = results['vacuumlevels']
        kvp['evac'] = (subresults['evac1'] + subresults['evac2']) / 2
        kd['evac'] = ('Vaccum level (PBE)', '', 'eV')
        kvp['evacdiff'] = subresults['evacdiff']
        kd['evacdiff'] = ('Vacuum level difference (PBE)', '', 'eV')
        kvp['dipz'] = subresults['dipz']
        kd['dipz'] = ('Dipole moment', '', '|e|Ang')

    # ------------------------ #

    # Currently all kvp are returned both as kvp and data XXX
    # Use results as data dict for now (change with new format) XXX
    return kvp, kd, results


def webpanel(row, key_descriptions):
    from asr.utils.custom import table

    # What about metals? XXX - should dos_at_ef be here?
    # There should be a metals check somewhere? XXX
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

    panel = ('PBE ground state', [[t]])

    return panel, []


# ---------- ASR globals and main ---------- #


group = 'property'
resources = '8:1h'  # How many resources are used? XXX
dependencies = ['asr.structureinfo', 'asr.gs']


if __name__ == '__main__':
    main()
