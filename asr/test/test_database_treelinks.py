from .materials import std_test_materials
import pytest


@pytest.mark.ci
def test_database_treelinks(asr_tmpdir, test_material):
    """Test asr.database.treelinks on a example defect tree."""
    import os
    from pathlib import Path
    from ase.io import write, read
    from asr.core import read_json
    from asr.setup.defects import main as setup_defects
    from asr.database.treelinks import main as treelinks
    from asr.database.material_fingerprint import (get_hash_of_atoms,
                                                   get_uid_of_atoms)

    write('unrelaxed.json', std_test_materials[1])
    p = Path('.')
    setup_defects(supercell=[3, 3, 1])

    # get material fingerprint for defect systems
    pathlist = list(p.glob('defects.*/charge_0'))
    for path in pathlist:
        os.system(f'cp {path.absolute()}/unrelaxed.json {path.absolute()}/structure.json')
        os.system(f'asr run asr.database.material_fingerprint {path.absolute()}')
    # get material fingerprint for pristine system
    pathlist = list(p.glob('defects.pristine_sc*'))
    for path in pathlist:
        os.system(f'cp {path.absolute()}/unrelaxed.json {path.absolute()}/structure.json')
        os.system(f'asr run asr.database.material_fingerprint {path.absolute()}')
    # get material fingerprint for host structure
    os.system('cp unrelaxed.json structure.json')
    os.system('asr run asr.database.material_fingerprint')

    # run asr.database.treelinks to create results and links.json files
    treelinks(include=['charge_0', 'defects.pristine_sc*'],
              exclude=[''])

    ref_uids = ["BN-d07bd84d0331",
                "B9N9-868d71d797c9",
                "B8N9-993df3e59cfb",
                "N8B9-3dbde4188cff",
                "B8N10-e3886951a993",
                "N8B10-e12d7c1775d2"]

    # test defect links
    pathlist = list(p.glob('defects.*/charge_0'))
    for path in pathlist:
        links = read_json(path / 'links.json')
        assert links['uids'] == ref_uids

    # test pristine links
    pathlist = list(p.glob('defects.*/charge_0'))
    for path in pathlist:
        links = read_json(path / 'links.json')
        assert links['uids'] == ref_uids
