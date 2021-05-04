from .materials import std_test_materials
import pytest


@pytest.mark.ci
def test_database_treelinks(asr_tmpdir):
    """Test asr.database.treelinks on a example defect tree."""
    from pathlib import Path
    from ase.io import write
    from asr.core import read_json, chdir
    from asr.database.treelinks import main as treelinks
    from asr.database.material_fingerprint import main as material_fingerprint

    # set up general folder structure to run treelinks recipe on
    for i, material in enumerate(std_test_materials):
        folderpath = Path(f'folder_{i}')
        folderpath.mkdir()
        write(folderpath / 'structure.json', material)
        with chdir(folderpath):
            material_fingerprint(atoms=material)
    write('structure.json', std_test_materials[0])
    material_fingerprint(atoms=std_test_materials[0])

    # run asr.database.treelinks to create results and links.json files
    treelinks(include=['folder_*'], exclude=[''])

    # define reference uids to compare to
    ref_uids = ["Si2-9552f5fb34d3",
                "BN-d07bd84d0331",
                "Ag-38f9b4cf2331",
                "Fe-551991cb0ca5"]
    ref_uids.sort()

    # test links and compare the created links to the reference links
    p = Path('.')
    pathlist = list(p.glob('folder_*'))
    for path in pathlist:
        links = read_json(path / 'links.json')
        uids = links['uids']
        uids.sort()
        for i, element in enumerate(links['uids']):
            assert element == ref_uids[i]
