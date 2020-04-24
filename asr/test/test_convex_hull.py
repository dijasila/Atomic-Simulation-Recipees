import pytest
from ase.db import connect
from ase.build import bulk


@pytest.mark.ci
def test_convex_hull(asr_tmpdir_w_params, mockgpaw):
    from ase.calculators.emt import EMT
    from asr.convex_hull import main
    metals = ['Al', 'Cu', 'Ag', 'Au', 'Ni',
              'Pd', 'Pt', 'C']

    energies = {}
    with connect('references.db') as db:
        for uid, element in enumerate(metals):
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

    metal = 'Ag'
    ag = bulk(metal)
    ag.write('structure.json')
    results = main(databases=['references.db'])
    assert results['hform'] == -energies[metal]
