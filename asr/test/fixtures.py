"""Module containing the implementations of all ASR pytest fixtures."""
import numpy as np
from ase.parallel import world, broadcast
from asr.core import write_json
from .materials import std_test_materials, BN
import pytest
import datetime
from _pytest.tmpdir import _mk_tmp
from pathlib import Path
from asr.core import get_cache
from asr.core.specification import construct_run_spec
from asr.core.root import Repository
from asr.core.record import Record
from asr.core.dependencies import Dependencies


@pytest.fixture()
def mockgpaw(monkeypatch):
    """Fixture that mocks up GPAW."""
    import sys

    monkeypatch.syspath_prepend(Path(__file__).parent.resolve() / "mocks")
    for module in list(sys.modules):
        if "gpaw" in module:
            sys.modules.pop(module)

    yield sys.path

    for module in list(sys.modules):
        if "gpaw" in module:
            sys.modules.pop(module)


@pytest.fixture(params=std_test_materials)
def test_material(request):
    """Fixture that returns an ase.Atoms object representing a std test material."""
    return request.param.copy()


VARIOUS_OBJECT_TYPES = [
    1,
    1.02,
    1 + 1e-15,
    1 + 1j,
    1e20,
    1e-17,
    "a",
    (1, "a"),
    [1, "a"],
    [1, (1, "abc", [1.0, ("a",)])],
    np.array([1.1, 2.0], float),
    BN,
    set(["a", 1, "2"]),
    Path("directory1/directory2/file.txt"),
    datetime.datetime.now(),
    Dependencies([]),
]


@pytest.fixture
def external_file(asr_tmpdir):
    from asr.core import ExternalFile

    filename = "somefile.txt"
    Path(filename).write_text("sometext")
    return ExternalFile.fromstr(filename)


@pytest.fixture(params=VARIOUS_OBJECT_TYPES)
def various_object_types(request):
    """Fixture that yield object of different relevant types."""
    return request.param


@pytest.fixture()
def asr_tmpdir(request, tmp_path_factory):
    """Create temp folder and change directory to that folder.

    A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    from ase.utils import workdir

    if world.rank == 0:
        path = _mk_tmp(request, tmp_path_factory)
    else:
        path = None
    path = broadcast(path)

    with workdir(path):
        if world.rank == 0:
            Repository.initialize(path)

        yield path


def _get_webcontent(dbname="database.db"):
    from asr.database.fromtree import main as fromtree

    # from asr.database.material_fingerprint import main as mf

    # mf()
    fromtree(recursive=True)
    content = ""
    from asr.database.app import App, get_project_from_database

    if world.rank == 0:
        dbapp = App()
        project = get_project_from_database(dbname)
        dbapp.add_project(project)
        dbapp.initialize()
        flask = dbapp.flask

        flask.testing = True
        with flask.test_client() as c:
            project = dbapp.projects["database.db"]
            db = project["database"]
            uid_key = project["uid_key"]
            row = db.get(id=1)
            uid = row.get(uid_key)
            url = f"/database.db/row/{uid}"
            content = c.get(url).data.decode()
            content = content.replace("\n", "").replace(" ", "")
    else:
        content = None
    content = broadcast(content)
    return content


@pytest.fixture(autouse=True)
def set_asr_test_environ_variable(monkeypatch):
    monkeypatch.setenv("ASRTESTENV", "true")


@pytest.fixture()
def get_webcontent():
    """Return a utility function that can create and return webcontent."""
    return _get_webcontent


@pytest.fixture()
def fast_calc():
    fast_calc = {
        "name": "gpaw",
        "kpts": {"density": 1, "gamma": True},
        "xc": "PBE",
    }
    return fast_calc


@pytest.fixture()
def asr_tmpdir_w_params(asr_tmpdir):
    """Make temp dir and create a params.json with settings for fast evaluation."""
    fast_calc = {
        "name": "gpaw",
        "kpts": {"density": 1, "gamma": True},
        "xc": "PBE",
    }
    params = {
        "asr.c2db.gs:calculate": {
            "calculator": fast_calc,
        },
        "asr.c2db.gs": {
            "calculator": fast_calc,
        },
        "asr.c2db.bandstructure": {
            "npoints": 10,
            "calculator": fast_calc,
        },
        "asr.hse": {
            "calculator": fast_calc,
            "kptdensity": 2,
        },
        "asr.c2db.gw:gs": {
            "kptdensity": 2,
        },
        "asr.bse:calculate": {
            "kptdensity": 2,
        },
        "asr.c2db.pdos:calculate": {
            "kptdensity": 2,
            "emptybands": 5,
        },
        "asr.c2db.piezoelectrictensor": {
            "calculator": fast_calc,
            "relaxcalculator": fast_calc,
        },
        "asr.c2db.formalpolarization": {
            "calculator": {
                "name": "gpaw",
                "kpts": {"density": 2},
            },
        },
    }

    write_json("params.json", params)


@pytest.fixture(params=std_test_materials)
def duplicates_test_db(request, asr_tmpdir):
    """Set up a database containing only duplicates of a material."""
    import ase.db

    db = ase.db.connect("duplicates.db")
    atoms = request.param.copy()

    db.write(atoms=atoms)

    rotated_atoms = atoms.copy()
    rotated_atoms.rotate(23, v="z", rotate_cell=True)
    db.write(atoms=rotated_atoms, magstate="FM")

    pbc_c = atoms.get_pbc()
    repeat = np.array([2, 2, 2], int)
    repeat[~pbc_c] = 1
    supercell_ref = atoms.repeat(repeat)
    db.write(supercell_ref)

    translated_atoms = atoms.copy()
    translated_atoms.translate(0.5)
    db.write(translated_atoms)

    rattled_atoms = atoms.copy()
    rattled_atoms.rattle(0.001)
    db.write(rattled_atoms)

    stretch_nonpbc_atoms = atoms.copy()
    cell = stretch_nonpbc_atoms.get_cell()
    pbc_c = atoms.get_pbc()
    cell[~pbc_c][:, ~pbc_c] *= 2
    stretch_nonpbc_atoms.set_cell(cell)
    db.write(stretch_nonpbc_atoms)

    return (atoms, db)


@pytest.fixture
def record(various_object_types):
    run_spec = construct_run_spec(
        name="asr.test",
        parameters={"a": 1},
        version=0,
    )
    run_record = Record(
        run_specification=run_spec,
        result=various_object_types,
    )
    return run_record


@pytest.fixture
def fscache(asr_tmpdir):
    cache = get_cache("filesystem")
    return cache


@pytest.fixture()
def crosslinks_test_dbs(asr_tmpdir):
    """Set up database for testing the crosslinks recipe."""
    from ase.io import write, read
    from ase.db import connect
    from asr.core import chdir
    from asr.database.material_fingerprint import main as material_fingerprint
    from asr.database.treelinks import main as treelinks
    from asr.database.fromtree import main as fromtree

    write("structure.json", std_test_materials[0])
    p = Path(".")

    for i in range(1, len(std_test_materials)):
        dirpath = Path(p / f"folder_{i - 1}")
        dirpath.mkdir()
        write(dirpath / "structure.json", std_test_materials[i])

    pathlist = list(p.glob("folder_*"))
    for path in pathlist:
        with chdir(path):
            material_fingerprint(atoms=read("structure.json"))
    material_fingerprint(atoms=read("structure.json"))

    # run asr.database.treelinks to create results and links.json files
    treelinks(include=["folder_*"], exclude=[""])

    # first, create one database based on the tree structure
    fromtree(recursive=True, dbname="db.db")
    # second, create one database with only the parent structure present
    fromtree(recursive=True, dbname="dbref.db")

    # set metadata in such a way that asr.database.crosslinks can work correctly
    db = connect("db.db")
    db.delete([1])
    dbref = connect("dbref.db")
    dbref.delete([2, 3, 4])
    db.metadata = {
        "title": "Example DB",
        "link_name": "{row.formula}-{row.uid}",
        "link_url": "test/test/{row.uid}",
    }
    dbref.metadata = {
        "title": "Example Reference DB",
        "link_name": "{row.uid}-{row.formula}",
        "link_url": "testref/testref/{row.uid}",
    }

    return None


@pytest.fixture
def database_with_one_row(asr_tmpdir):
    from ase.db import connect
    from asr.test.materials import Ag
    database = connect("test_database.db")
    database.write(Ag)
    return database
