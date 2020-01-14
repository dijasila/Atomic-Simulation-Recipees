from asr.core import command, argument


key_descriptions = {
    "berry": {"Topology": "KVP: Band topology (Topology)"},
    "bse": {"E_B": "KVP: BSE binding energy (Exc. bind. energy) [eV]"},
    "convex_hull": {
        "ehull": "KVP: Energy above convex hull [eV/atom]",
        "hform": "KVP: Heat of formation [eV/atom]",
        "thermodynamic_stability_level": "KVP: Thermodynamic stability level",
    },
    "gs": {
        "forces": "Forces on atoms [eV/Angstrom]",
        "stresses": "Stress on unit cell [eV/Angstrom^(dim-1)]",
        "etot": "KVP: Total energy (Tot. En.) [eV]",
        "evac": "KVP: Vacuum level (Vacuum level) [eV]",
        "evacdiff": "KVP: Vacuum level shift (Vacuum level shift) [eV]",
        "dipz": "KVP: Out-of-plane dipole [e * Ang]",
        "efermi": "KVP: Fermi level (Fermi level) [eV]",
        "gap": "KVP: Band gap (Band gap) [eV]",
        "vbm": "KVP: Valence band maximum (Val. band max.) [eV]",
        "cbm": "KVP: Conduction band minimum (Cond. band max.) [eV]",
        "gap_dir": "KVP: Direct band gap (Dir. band gap) [eV]",
        "vbm_dir": (
            "KVP: Direct valence band maximum (Dir. val. band max.) [eV]"
        ),
        "cbm_dir": (
            "KVP: Direct conduction band minimum (Dir. cond. band max.) [eV]"
        ),
        "gap_dir_nosoc": (
            "KVP: Direct gap without SOC (Dir. gap wo. soc.) [eV]"
        ),
    },
    "gw": {
        "vbm_gw_nosoc": "GW valence band max. w/o soc [eV]",
        "cbm_gw_nosoc": "GW condution band min. w/o soc [eV]",
        "dir_gap_gw_nosoc": "GW direct gap w/o soc [eV]",
        "gap_gw_nosoc": "GW gap w/o soc [eV]",
        "kvbm_nosoc": "k-point of GW valence band max. w/o soc",
        "kcbm_nosoc": "k-point of GW conduction band min. w/o soc",
        "vbm_gw": "KVP: GW valence band max. [eV]",
        "cbm_gw": "KVP: GW conduction band min. [eV]",
        "dir_gap_gw": "KVP: GW direct gap [eV]",
        "gap_gw": "KVP: GW gap [eV]",
        "kvbm": "k-point of GW valence band max.",
        "kcbm": "k-point of GW conduction band min.",
        "efermi_gw_nosoc": "GW Fermi energy w/o soc [eV]",
        "efermi_gw_soc": "GW Fermi energy [eV]",
    },
    "hse": {
        "vbm_hse_nosoc": "HSE valence band max. w/o soc [eV]",
        "cbm_hse_nosoc": "HSE condution band min. w/o soc [eV]",
        "dir_gap_hse_nosoc": "HSE direct gap w/o soc [eV]",
        "gap_hse_nosoc": "HSE gap w/o soc [eV]",
        "kvbm_nosoc": "k-point of HSE valence band max. w/o soc",
        "kcbm_nosoc": "k-point of HSE conduction band min. w/o soc",
        "vbm_hse": "KVP: HSE valence band max. [eV]",
        "cbm_hse": "KVP: HSE conduction band min. [eV]",
        "dir_gap_hse": "KVP: HSE direct gap [eV]",
        "gap_hse": "KVP: HSE gap [eV]",
        "kvbm": "k-point of HSE valence band max.",
        "kcbm": "k-point of HSE conduction band min.",
        "efermi_hse_nosoc": "HSE Fermi energy w/o soc [eV]",
        "efermi_hse_soc": "HSE Fermi energy [eV]",
    },
    "infraredpolarizability": {
        "alphax_lat": "KVP: Static ionic polarizability, x-direction [Ang]",
        "alphay_lat": "KVP: Static ionic polarizability, y-direction [Ang]",
        "alphaz_lat": "KVP: Static ionic polarizability, z-direction [Ang]",
        "alphax": "KVP: Static total polarizability, x-direction [Ang]",
        "alphay": "KVP: Static total polarizability, y-direction [Ang]",
        "alphaz": "KVP: Static total polarizability, z-direction [Ang]",
    },
    "magnetic_anisotropy": {
        "spin_axis": "KVP: Suggested spin direction for SOC",
        "E_x": "KVP: SOC total energy difference in x-direction",
        "E_y": "KVP: SOC total energy difference in y-direction",
        "E_z": "KVP: SOC total energy difference in z-direction",
        "theta": "Spin direction, theta, polar coordinates [radians]",
        "phi": "Spin direction, phi, polar coordinates [radians]",
        "dE_zx": (
            "KVP: Magnetic anisotropy energy "
            "(zx-component) [meV/formula unit]"
        ),
        "dE_zy": (
            "KVP: Magnetic anisotropy energy "
            "(zy-component) [meV/formula unit]"
        ),
    },
    "pdos": {
        "pdos_nosoc":
        "Projected density of states without spin-orbit coupling (PDOS no "
        "soc)",
        "pdos_soc":
        "Projected density of states with spin-orbit coupling (PDOS w. soc)",
        "dos_at_ef_nosoc":
        "KVP: Density of states at the Fermi energy without spin-orbit "
        "coupling (DOS at ef no soc) [states/eV]",
        "dos_at_ef_soc":
        "KVP: Density of states at the Fermi energy with spin-orbit coupling"
        " (DOS at ef w. soc) [states/eV]",
    },
    "phonons": {
        "minhessianeig": "KVP: Minimum eigenvalue of Hessian [eV/Ang^2]",
        "dynamic_stability_level": "KVP: Dynamic stability level",
    },
    "plasmafrequency": {
        "plasmafreq_vv": "Plasma frequency tensor [Hartree]",
        "plasmafrequency_x": "KVP: 2D plasma frequency, x-direction "
        "[eV/Ang^0.5] #2D",
        "plasmafrequency_y": "KVP: 2D plasma frequency, y-direction "
        "[eV/Ang^0.5] #2D",
    },
    "polarizability": {
        "alphax_el": "KVP: Static electronic polarizability,"
        " x-direction [Ang]",
        "alphay_el": "KVP: Static electronic polarizability,"
        " y-direction [Ang]",
        "alphaz_el": "KVP: Static electronic polarizability,"
        " z-direction [Ang]",
    },
    "relax": {
        "etot": "Total energy [eV]",
        "edft": "DFT total energy [eV]",
        "spos": "Array: Scaled positions",
        "symbols": "Array: Chemical symbols",
        "a": "Cell parameter a [Ang]",
        "b": "Cell parameter b [Ang]",
        "c": "Cell parameter c [Ang]",
        "alpha": "Cell parameter alpha [deg]",
        "beta": "Cell parameter beta [deg]",
        "gamma": "Cell parameter gamma [deg]",
    },
    "stiffness": {
        "c_11": "KVP: Stiffness tensor: 11-component [N/m^(dim-1)]",
        "c_22": "KVP: Stiffness tensor: 22-component [N/m^(dim-1)]",
        "c_33": "KVP: Stiffness tensor: 33-component [N/m^(dim-1)]",
        "c_23": "KVP: Stiffness tensor: 23-component [N/m^(dim-1)]",
        "c_13": "KVP: Stiffness tensor: 13-component [N/m^(dim-1)]",
        "c_12": "KVP: Stiffness tensor: 12-component [N/m^(dim-1)]",
        "speed_of_sound_x": "KVP: Speed of sound in x direction [m/s]",
        "speed_of_sound_y": "KVP: Speed of sound in y direction [m/s]",
        "stiffness_tensor": ("Stiffness tensor [N/m^dim]"),
    },
    "structureinfo": {
        "magstate": "KVP: Magnetic state",
        "is_magnetic": "KVP: Material is magnetic (Magnetic)",
        "cell_area": "KVP: Area of unit-cell [Ang^2]",
        "has_invsymm": "KVP: Inversion symmetry",
        "stoichiometry": "KVP: Stoichiometry",
        "spacegroup": "KVP: Space group",
        "spgnum": "KVP: Space group number",
        "crystal_prototype": "KVP: Crystal prototype",
    },
    "database.material_fingerprint": {
        'asr_id': 'KVP: Material fingerprint',
        'uid': 'KVP: Unique identifier'
    },
    "info.json": {
        'class': 'KVP: Material class',
        'doi': 'KVP: Monolayer DOI',
        'icsd_id': 'KVP: ICSD id of parent bulk structure',
        'cod_id': 'KVP: COD id of parent bulk structure'
    },
    "emasses": {
        'emass_vb_dir1':
        'KVP: Valence band effective mass in direction 1 [m_e]',
        'emass_vb_dir2':
        'KVP: Valence band effective mass in direction 2 [m_e]',
        'emass_vb_dir3':
        'KVP: Valence band effective mass in direction 3 [m_e]',
        'emass_cb_dir1':
        'KVP: Conduction band effective mass in direction 1 [m_e]',
        'emass_cb_dir2':
        'KVP: Conduction band effective mass in direction 2 [m_e]',
        'emass_cb_dir3':
        'KVP: Conduction band effective mass in direction 3 [m_e]',
    }
}


@command()
@argument('database')
def main(database):
    """Analyze database and set metadata.

    This recipe loops through all rows in a database and figures out what keys
    are present in the database. It also figures our what kind of materials
    (1D, 2D, 3D) are in the database. Then it saves those values in the
    database metadata like:

    ase db -m database.db

    {
    ...
    'key_value_pairs': ['etot', 'gap', ...],
    # List of keys for key-value-pairs contained in DB.
    ...
    }
    """
    from ase.db import connect

    db = connect(database)

    print('Row #')
    keys = set()
    for ir, row in enumerate(db.select(include_data=False)):
        if ir % 100 == 0:
            print(ir)
        print(row.key_value_pairs.keys())
        keys.update(set(row.key_value_pairs.keys()))

    metadata = db.metadata
    metadata.update({'keys': sorted(list(keys))})
    db.metadata = metadata


if __name__ == '__main__':
    main.cli()
