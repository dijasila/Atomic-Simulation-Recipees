from asr.core import command, argument, get_recipes, ASRResult
from asr.dimensionality import get_dimtypes
from asr.c2db.labels import label_explanation
from ase.db.core import KeyDescription as ASEKeyDescription


class ASRKeyDescription(ASEKeyDescription):
    def __init__(self, *args, iskvp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.iskvp = iskvp


def kd(key, long=None, short=None, unit='', iskvp=False):
    if short is None:
        short = long
    return ASRKeyDescription(key, shortdesc=short, longdesc=long,
                             unit=unit, iskvp=iskvp)


def kvp(*args, **kwargs):
    return kd(*args, **kwargs, iskvp=True)

# Style: "KVP: Long description !short description! [unit]


def all_the_keydescs():
    yield 'berry', [
        kvp('Topology', 'Band topology', 'Topology'),
        kd('phi0_km', 'Berry phase spectrum localized in k0'),
        kd('s0_km', 'Spin of berry phases localized in k0'),
        kd('phi1_km', 'Berry phase spectrum localized in k1'),
        kd('s1_km', 'Spin of berry phases localized in k1'),
        kd('phi2_km', 'Berry phase spectrum localized in k2'),
        kd('s2_km', 'Spin of berry phases localized in k3'),
        kd('phi0_pi_km', 'Berry phase spectrum at k2=pi localized in k0'),
        kd('s0_pi_km', 'Spin of berry at phases at k2=pi localized in k0'),
    ]
    yield 'bse', [
        kvp('E_B', 'Exciton binding energy from BSE ',
            'Exc. bind. energy', unit='eV'),
    ]
    yield 'convex_hull', [
        kvp('ehull', 'Energy above convex hull', unit='eV/atom'),
        kvp('hform', 'Heat of formation', unit='eV/atom'),
        kvp('thermodynamic_stability_level', 'Thermodynamic stability level'),
    ]
    yield 'defectinfo', [
        kvp('defect_name', 'Defect name {type}_{position}', 'Defect name'),
        kvp('host_name', 'Host formula'),
        kvp('charge_state', 'Charge state'),
    ]
    yield 'defect_symmetry', [
        kvp('defect_pointgroup', 'Defect point group', 'Point group'),
    ]
    yield 'magstate', [
        kvp('magstate', 'Magnetic state'),
        kvp('is_magnetic', 'Material is magnetic', 'Magnetic'),
        kvp('nspins', 'Number of spins in calculator', 'n-spins'),
    ]
    yield 'gs', [
        kd('forces', 'Forces on atoms', unit='eV/Å'),
        kd('stresses', 'Stress on unit cell', unit='`eV/Å^{dim-1}`'),
        kvp('etot', 'Total energy', 'Tot. En.', unit='eV'),
        kvp('evac', 'Vacuum level', 'Vacuum level', 'eV'),
        kvp('evacdiff', 'Vacuum level difference', unit='eV'),
        kvp('dipz', 'Out-of-plane dipole along +z axis',
            unit='e · Å/unit cell'),
        kvp('efermi', 'Fermi level', unit='eV'),
        kvp('gap', 'Band gap', unit='eV'),
        kvp('vbm', 'Valence band maximum', 'Val. band maximum', unit='eV'),
        kvp('cbm', 'Conduction band minimum', 'Cond. band minimum', unit='eV'),
        kvp('gap_dir', 'Direct band gap', 'Dir. band gap', 'eV'),
        kvp('gap_dir_nosoc', 'Direct gap w/o soc.',
            'Dir. gap wo. soc.', 'eV'),
        kvp('gap_nosoc', 'Gap w/o soc.', 'Gap wo. soc.', 'eV'),
        kvp('workfunction', 'Work function (avg. if finite dipole)',
            unit='eV'),
    ],
    yield 'gw', [
        kd('vbm_gw_nosoc', 'Valence band maximum w/o soc. (G₀W₀)', unit='eV'),
        kd('cbm_gw_nosoc', 'Conduction band minimum w/o soc. (G₀W₀)',
           unit='eV'),
        kd('gap_dir_gw_nosoc', 'Direct gap w/o soc. (G₀W₀)', unit='eV'),
        kd('gap_gw_nosoc', 'Gap w/o soc. (G₀W₀)', unit='eV'),
        kd('kvbm_nosoc', 'k-point of G₀W₀ valence band maximum w/o soc'),
        kd('kcbm_nosoc', 'k-point of G₀W₀ conduction band minimum w/o soc'),
        kvp('vbm_gw', 'Valence band maximum (G₀W₀)', unit='eV'),
        kvp('cbm_gw', 'Conduction band minimum (G₀W₀)', unit='eV'),
        kvp('gap_dir_gw', 'Direct band gap (G₀W₀)', unit='eV'),
        kvp('gap_gw', 'Band gap (G₀W₀)', unit='eV'),
        kd('kvbm', 'k-point of G₀W₀ valence band maximum'),
        kd('kcbm', 'k-point of G₀W₀ conduction band minimum'),
        kd('efermi_gw_nosoc', 'Fermi level w/o soc. (G₀W₀)', unit='eV'),
        kd('efermi_gw_soc', 'Fermi level (G₀W₀)', unit='eV'),
    ]
    yield 'hse', [
        kd('vbm_hse_nosoc', 'Valence band maximum w/o soc. (HSE06)', unit='eV'),
        kd('cbm_hse_nosoc', 'Conduction band minimum w/o soc. (HSE06)',
           unit='eV'),
        kd('gap_dir_hse_nosoc', 'Direct gap w/o soc. (HSE06) [eV]'),
        kd('gap_hse_nosoc', 'Band gap w/o soc. (HSE06) [eV]'),
        kd('kvbm_nosoc', 'k-point of HSE06 valence band maximum w/o soc'),
        kd('kcbm_nosoc', 'k-point of HSE06 conduction band minimum w/o soc'),
        kvp('vbm_hse', 'Valence band maximum (HSE06)', unit='eV'),
        kvp('cbm_hse', 'Conduction band minimum (HSE06)', unit='eV'),
        kvp('gap_dir_hse', 'Direct band gap (HSE06)', unit='eV'),
        kvp('gap_hse', 'Band gap (HSE06)', unit='eV'),
        kd('kvbm', 'k-point of HSE06 valence band maximum'),
        kd('kcbm', 'k-point of HSE06 conduction band minimum'),
        kd('efermi_hse_nosoc', 'Fermi level w/o soc. (HSE06)', unit='eV'),
        kd('efermi_hse_soc', 'Fermi level (HSE06) [eV]'),
    ]
    yield 'infraredpolarizability', [
        kvp('alphax_lat', 'Lattice polarizability (x)', unit='Å'),
        kvp('alphay_lat', 'Lattice polarizability (y)', unit='Å'),
        kvp('alphaz_lat', 'Lattice polarizability (z)', unit='Å'),
        kvp('alphax', 'Static polarizability (x)', unit='Å'),
        kvp('alphay', 'Static polarizability (y)', unit='Å'),
        kvp('alphaz', 'Static polarizability (z)', unit='Å'),
    ]
    yield 'magnetic_anisotropy', [
        kvp('spin_axis', 'Magnetic easy axis'),
        kvp('E_x', 'Soc. total energy, x-direction', unit='meV/unit cell'),
        kvp('E_y', 'Soc. total energy, y-direction', unit='meV/unit cell'),
        kvp('E_z', 'Soc. total energy, z-direction', unit='meV/unit cell'),
        kd('theta', 'Easy axis, polar coordinates, theta', unit='radians'),
        kd('phi', 'Easy axis, polar coordinates, phi', unit='radians'),
        kvp('dE_zx', 'Magnetic anisotropy (E<sub>z</sub> - E<sub>x</sub>)',
            unit='meV/unit cell'),
        kvp('dE_zy', 'Magnetic anisotropy (E<sub>z</sub> - E<sub>y</sub>)',
            unit='meV/unit cell'),
    ]
    yield 'exchange', [
        kvp('J', 'Nearest neighbor exchange coupling', unit='meV'),
        kvp('A', 'Single-ion anisotropy (out-of-plane)', unit='meV'),
        kvp('lam', 'Anisotropic exchange (out-of-plane)', unit='meV'),
        kvp('spin', 'Maximum value of S_z at magnetic sites'),
        kvp('N_nn', 'Number of nearest neighbors'),
    ]
    yield 'collect_spiral', [
        kvp('bandwidth', 'Spin spiral bandwidth', unit='meV'),
        kd('Qmin', 'Magnetic order vector Q', unit='inverse unit cell'),
    ]
    yield 'spinorbit', [
        kvp('soc_bw', 'Bandwidth of SOC energies', unit='meV'),
        kvp('theta_min', 'Angles from z->x', unit='deg'),
        kvp('phi_min', 'Angles from x->y', unit='deg'),
        kvp('projected_soc', 'Projected SOC for spin spirals')
    ]
    yield 'dmi', [
        kvp('DMI', 'Dzyaloshinskii Moriya interaction', unit='meV Å')
    ]
    yield 'orbmag', [
        kvp('orbmag_sum', 'Sum of local orbital magnetic moments', unit='μ_B'),
        kvp('orbmag_max', 'Maximum norm of local orbital magnetic moments',
            unit='μ_B'),
    ]
    yield 'pdos', [
        kd('pdos_nosoc', 'Projected density of states w/o soc.',
           'PDOS no soc'),
        kd('pdos_soc', 'Projected density of states', 'PDOS'),
        kvp('dos_at_ef_nosoc', 'Density of states at the Fermi level w/o soc.',
            'DOS at ef no soc.', unit='states/(eV · unit cell)'),
        kvp('dos_at_ef_soc', 'Density of states at the Fermi level',
            'DOS at ef', unit='states/(eV · unit cell)'),
    ]
    yield 'phonons', [
        kvp('minhessianeig', 'Minimum eigenvalue of Hessian', unit='eV/Å²'),
        kvp('dynamic_stability_phonons',
            'Phonon dynamic stability (low/high)'),
    ]
    yield 'plasmafrequency', [
        kd('plasmafreq_vv', 'Plasma frequency tensor', unit='Hartree'),
        kvp('plasmafrequency_x', '2D plasma frequency (x)', unit='`eV/Å^0.5`'),
        kvp('plasmafrequency_y', '2D plasma frequency (y)', unit='`eV/Å^0.5`'),
    ]
    yield 'polarizability', [
        kvp('alphax_el', 'Optical polarizability (x)', unit='Å'),
        kvp('alphay_el', 'Optical polarizability (y)', unit='Å'),
        kvp('alphaz_el', 'Optical polarizability (z)', unit='Å'),
    ]
    yield 'relax', [
        kd('edft', 'DFT total enrgy', unit='eV'),
        kd('spos', 'Array: Scaled positions'),
        kd('symbols', 'Array: Chemical symbols'),
        kd('a', 'Cell parameter a', unit='Å'),
        kd('b', 'Cell parameter b', unit='Å'),
        kd('c', 'Cell parameter c', unit='Å'),
        kd('alpha', 'Cell parameter alpha', unit='deg'),
        kd('beta', 'Cell parameter beta', unit='deg'),
        kd('gamma', 'Cell parameter gamma', unit='deg'),
    ]

    stiffness_kvps = [
        kvp('speed_of_sound_x', 'Speed of sound (x)', unit='m/s'),
        kvp('speed_of_sound_y', 'Speed of sound (y)', unit='m/s'),
        kd('stiffness_tensor', 'Stiffness tensor', unit='`N/m^{dim-1}`'),
        kvp('dynamic_stability_stiffness',
            'Stiffness dynamic stability (low/high)'),
    ]
    for i in range(6):
        for j in range(6):
            stiffness_kvps.append(kvp(
                f'c_{i}{j}', f'Stiffness tensor, {i}{j}-component',
                unit='`N/m^{dim-1}`'))

    yield 'stiffness', stiffness_kvps

    yield 'structureinfo', [
        kvp('cell_area', 'Area of unit-cell [`Å²`]'),
        kvp('has_inversion_symmetry', 'Material has inversion symmetry'),
        kvp('stoichiometry', 'Stoichiometry'),
        kvp('spacegroup', 'Space group (AA stacking)'),
        kvp('spgnum', 'Space group number (AA stacking)'),
        kvp('layergroup', 'Layer group'),
        kvp('lgnum', 'Layer group number'),
        kvp('pointgroup', 'Point group'),
        kvp('crystal_type', 'Crystal type'),
    ]
    yield 'database.material_fingerprint', [
        kvp('asr_id', 'Material unique ID'),
        kvp('uid', 'Unique identifier'),
    ]
    dimensionality_kvps = [
        kvp('dim_primary', 'Dim. with max. scoring parameter'),
        kvp('dim_primary_score',
            'Dimensionality scoring parameter of primary dimensionality.'),
        kvp('dim_nclusters_0D', 'Number of 0D clusters.'),
        kvp('dim_nclusters_1D', 'Number of 1D clusters.'),
        kvp('dim_nclusters_2D', 'Number of 2D clusters.'),
        kvp('dim_nclusters_3D', 'Number of 3D clusters.'),
        kvp('dim_threshold_0D', '0D dimensionality threshold.'),
        kvp('dim_threshold_1D', '1D dimensionality threshold.'),
        kvp('dim_threshold_2D', '2D dimensionality threshold.'),
        kvp('dim_threshold_3D', '3D dimensionality threshold.'),
    ]

    dimensionality_kvps += [
        kvp(f'dim_score_{dimtype}',
            f'Dimensionality score of dimtype={dimtype}')
        for dimtype in get_dimtypes()]

    yield 'dimensionality', dimensionality_kvps

    yield 'setinfo', [
        kvp('first_class_material',
            'A first class material marks a physical material.',
            'First class material', unit='bool'),
    ]
    yield 'info.json', [
        kvp('class', 'Material class'),
        kvp('doi', 'Monolayer reported DOI'),
        kvp('icsd_id', 'ICSD id of parent bulk structure'),
        kvp('cod_id', 'COD id of parent bulk structure'),
    ]
    yield 'emasses', [
        kvp('emass_vb_dir1', 'Valence band effective mass, direction 1',
            unit='`m_e`'),
        kvp('emass_vb_dir2', 'Valence band effective mass, direction 2',
            unit='`m_e`'),
        kvp('emass_vb_dir3', 'Valence band effective mass, direction 3',
            unit='`m_e`'),
        kvp('emass_cb_dir1', 'Conduction band effective mass, direction 1',
            unit='`m_e`'),
        kvp('emass_cb_dir2', 'Conduction band effective mass, direction 2',
            unit='`m_e`'),
        kvp('emass_cb_dir3', 'Conduction band effective mass, direction 3',
            unit='`m_e`'),
    ]
    yield 'c2db.labels', [
        kvp('origin', label_explanation),
    ]
    yield 'database.fromtree', [
        kvp('folder', 'Path to collection folder'),
    ]

    piezo_kvps = []
    for i in range(1, 4):
        for j in range(1, 7):
            key = 'e_{}{}'.format(i, j)
            name = 'Piezoelectric tensor'
            description = f'Piezoelectric tensor {i}{j}-component'
            unit = '`\\text{Å}^{-1}`'
            piezo_kvps.append(kvp(key, name, description, unit=unit))

    yield 'piezoelectrictensor', piezo_kvps

    # Key descriptions like has_asr_gs_calculate
    extras = []
    for recipe in get_recipes():
        key = 'has_' + recipe.name.replace('.', '_').replace('@', '_')
        extras.append(kd(key, f'{recipe.name} is calculated'))

    # Additional key description for spin coherence times of QPOD
    extras.append(kd('sct', 'Spin coherence time T2', unit='ms'))
    yield 'extra', extras


def get_all_keydescs_as_dict():
    all_keydescs = {}
    for section, keydescs in all_the_keydescs():
        assert section not in all_keydescs
        all_keydescs[section] = {keydesc.key: keydesc for keydesc in keydescs}
    return all_keydescs


key_descriptions = get_all_keydescs_as_dict()


@command()
@argument('database', type=str)
def main(database: str) -> ASRResult:
    """Analyze database and set metadata.

    This recipe loops through all rows in a database and figures out what keys
    are present in the database. It also figures our what kind of materials
    (1D, 2D, 3D) are in the database. Then it saves those values in the
    database metadata under {'keys': ['etot', 'gap', ...]}
    """
    from ase.db import connect

    db = connect(database)

    print('Row #')
    keys = set()
    for ir, row in enumerate(db.select(include_data=False)):
        if ir % 100 == 0:
            print(ir)
        keys.update(set(row.key_value_pairs.keys()))

    metadata = db.metadata
    metadata.update({'keys': sorted(list(keys))})
    db.metadata = metadata


if __name__ == '__main__':
    main.cli()
