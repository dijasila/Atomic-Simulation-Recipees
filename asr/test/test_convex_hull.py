import pytest
from ase.db import connect
from ase.build import bulk


metal_alloys = ['Ag', 'Au', 'Ag,Au', 'Ag,Au,Al']


@pytest.fixture()
def refdb(asr_tmpdir_w_params):
    from ase.calculators.emt import EMT
    elemental_metals = ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                        'Pd', 'Pt', 'C']

    energies = {}
    with connect('references.db') as db:
        for uid, element in enumerate(elemental_metals):
            atoms = bulk(element)
            atoms.calc = EMT()
            en = atoms.get_potential_energy()
            energies[element] = en
            db.write(atoms, uid=uid, etot=en)

    metadata = {'title': 'Metal references',
                'legend': 'Metals',
                'name': '{row.formula}',
                'link': 'NOLINK',
                'label': '{row.formula}',
                'method': 'DFT'}
    db.metadata = metadata

    return db, 'references.db', energies


@pytest.mark.xfail(reason='TODO')
@pytest.mark.ci
@pytest.mark.parametrize('metals', metal_alloys)
@pytest.mark.parametrize('energy_key', [None, 'etot'])
def test_convex_hull(refdb, mockgpaw, get_webcontent,
                     metals, energy_key, fast_calc):
    from asr.c2db.convex_hull import main
    db, dbname, energies = refdb

    metadata = db.metadata
    if energy_key is not None:
        metadata['energy_key'] = energy_key
        db.metadata = metadata

    metal_atoms = metals.split(',')
    nmetalatoms = len(metal_atoms)
    atoms = bulk(metal_atoms[0])
    atoms = atoms.repeat((1, 1, nmetalatoms))
    atoms.set_chemical_symbols(metal_atoms)

    results = main(
        atoms=atoms,
        databases=['references.db'],
        calculator=fast_calc,
    )
    assert results['hform'] == -sum(energies[element]
                                    for element in metal_atoms) / nmetalatoms

    atoms.write('structure.json')
    get_webcontent()


def make_alloy(commasepmetals):

    metal_atoms = commasepmetals.split(',')
    nmetalatoms = len(metal_atoms)
    atoms = bulk('Ag')
    atoms = atoms.repeat((1, 1, nmetalatoms))
    atoms.set_chemical_symbols(metal_atoms)
    return atoms


@pytest.fixture()
def refdbwithalloys(refdb):
    from ase.calculators.emt import EMT
    elemental_metals = ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                        'Pd', 'Pt', 'C']

    db, dbname, energies = refdb
    alloys = [
        ','.join([metal1, metal2])
        for metal1 in elemental_metals
        for metal2 in elemental_metals
    ]

    energies = {}
    with connect('references_alloys.db') as db:
        for uid, alloy in enumerate(alloys):
            atoms = make_alloy(alloy)
            atoms.calc = EMT()
            en = atoms.get_potential_energy()
            energies[alloy] = en
            db.write(atoms, uid=uid, etot=en)

    metadata = {'title': 'Metal references',
                'legend': 'Alloys',
                'name': '{row.formula}',
                'link': 'NOLINK',
                'label': '{row.formula}',
                'method': 'DFT'}
    db.metadata = metadata

    return db, dbname, 'references_alloys.db', energies


@pytest.mark.xfail(reason='TODO')
@pytest.mark.ci
@pytest.mark.parametrize('alloy', ['Ag,Au,Al', 'Ag,Al'])
def test_convex_hull_with_two_reference_databases(
        refdbwithalloys, mockgpaw, get_webcontent, alloy, fast_calc):
    from asr.c2db.convex_hull import main
    db, dbname, alloydbname, energies = refdbwithalloys

    atoms = make_alloy(alloy)
    atoms.write('structure.json')
    main(
        atoms=atoms,
        databases=[dbname, alloydbname],
        calculator=fast_calc,
    )
    get_webcontent()
