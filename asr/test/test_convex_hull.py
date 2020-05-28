import pytest
from ase.db import connect
from ase.build import bulk


metal_alloys = ['Ag', 'Au', 'Ag,Au', 'Ag,Au,Al']


@pytest.mark.ci
@pytest.mark.parametrize('metals', metal_alloys)
def test_convex_hull(asr_tmpdir_w_params, mockgpaw, get_webcontent,
                     metals):
    from ase.calculators.emt import EMT
    from asr.convex_hull import main
    elemental_metals = ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                        'Pd', 'Pt', 'C']

    energies = {}
    with connect('references.db') as db:
        for uid, element in enumerate(elemental_metals):
            atoms = bulk(element)
            atoms.set_calculator(EMT())
            en = atoms.get_potential_energy()
            energies[element] = en
            db.write(atoms, uid=uid)

    db.metadata = {'title': 'Metal references',
                   'legend': 'Metals',
                   'name': '{row.formula}',
                   'link': 'NOLINK',
                   'label': '{row.formula}',
                   'method': 'DFT'}

    metal_atoms = metals.split(',')
    nmetalatoms = len(metal_atoms)
    atoms = bulk(metal_atoms[0])
    atoms = atoms.repeat((1, 1, nmetalatoms))
    atoms.set_chemical_symbols(metal_atoms)
    atoms.write('structure.json')
    results = main(databases=['references.db'])
    assert results['hform'] == -sum(energies[element]
                                    for element in metal_atoms) / nmetalatoms
    get_webcontent()
