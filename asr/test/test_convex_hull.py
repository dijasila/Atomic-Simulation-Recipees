import pytest
from ase.db import connect
from ase.build import bulk
from asr.c2db.convex_hull import main


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


@pytest.mark.ci
@pytest.mark.parametrize('metals', metal_alloys)
@pytest.mark.parametrize('energy_key', [None, 'etot'])
def test_convex_hull(refdb, get_webcontent,
                     metals, energy_key):
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

    energy = 0.0

    results = main(
        formula=atoms.symbols.formula,
        energy=energy,
        databases=['references.db'],
    )
    hform = -sum(energies[element] for element in metal_atoms) / nmetalatoms
    assert results['hform'] == pytest.approx(hform)

    # atoms.write('structure.json')
    # get_webcontent()


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


@pytest.mark.ci
@pytest.mark.parametrize('alloy', ['Ag,Au,Al', 'Ag,Al'])
def test_convex_hull_with_two_reference_databases(
        refdbwithalloys, get_webcontent, alloy):
    db, dbname, alloydbname, energies = refdbwithalloys

    atoms = make_alloy(alloy)
    atoms.write('structure.json')
    main(
        energy=0.0,
        formula=atoms.symbols.formula,
        databases=[dbname, alloydbname])
    # get_webcontent()
