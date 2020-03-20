from .conftest import BN, get_webcontent
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("atoms", [BN])
def test_piezoelectrictensor(separate_folder, mockgpaw, atoms):
    from ase.io import write
    from asr.piezoelectrictensor import main
    write('structure.json', atoms)
    main()
    content = get_webcontent('database.db')

    assert "Piezoelectrictensor" in content, content
    assert False


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
@pytest.mark.parametrize("atoms", [BN])
def test_piezoelectrictensor_BN_gpaw(separate_folder, atoms):
    from asr.relax import main as relax
    from asr.piezoelectrictensor import main as piezo
    from asr.setup.params import main as setupparams
    BN.write('unrelaxed.json')

    params = {
        'asr.relax': {
            'calculator': {
                'mode': {
                    'name': 'pw',
                    'ecut': 350
                },
                'kpts': {
                    'density': 1
                },
                None: None
            }
        },
        'asr.gs@calculate': {
            'calculator': {
                'mode': {
                    'name': 'pw',
                    'ecut': 350
                },
                'kpts': {
                    'density': 1
                },
                None: None
            }
        }
    }
    relax()
    setupparams(params=params)
    results = piezo()

    eps_vvv = results['eps_vvv']
    assert eps_vvv[0, 0, 0] == 3
