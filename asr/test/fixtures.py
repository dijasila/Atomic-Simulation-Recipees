"""Module containing the implementations of all ASR pytest fixtures."""

from ase.parallel import world, broadcast
from asr.core import write_json
from .materials import std_test_materials
import os
import pytest
from _pytest.tmpdir import _mk_tmp
from pathlib import Path


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
    return request.param


@pytest.fixture()
def asr_tmpdir(request, tmp_path_factory):
    """Create temp folder and change directory to that folder.

    A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    if world.rank == 0:
        path = _mk_tmp(request, tmp_path_factory)
    else:
        path = None
    path = broadcast(path)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def _get_webcontent(name='database.db'):
    from asr.database.fromtree import main as fromtree
    from asr.database.material_fingerprint import main as mf
    mf()
    fromtree()
    content = ""
    from asr.database import app as appmodule
    from pathlib import Path
    if world.rank == 0:
        from asr.database.app import app, initialize_project, projects

        tmpdir = Path("tmp/")
        tmpdir.mkdir()
        appmodule.tmpdir = tmpdir
        initialize_project(name)

        app.testing = True
        with app.test_client() as c:
            project = projects["database.db"]
            db = project["database"]
            uid_key = project["uid_key"]
            row = db.get(id=1)
            uid = row.get(uid_key)
            url = f"/database.db/row/{uid}"
            content = c.get(url).data.decode()
            content = (
                content
                .replace("\n", "")
                .replace(" ", "")
            )
    else:
        content = None
    content = broadcast(content)
    return content


@pytest.fixture(autouse=True)
def set_asr_test_environ_variable(monkeypatch):
    monkeypatch.setenv("ASRTESTENV", True)


@pytest.fixture()
def get_webcontent():
    """Return a utility function that can create and return webcontent."""
    return _get_webcontent


@pytest.fixture()
def asr_tmpdir_w_params(asr_tmpdir):
    """Make temp dir and create a params.json with settings for fast evaluation."""
    params = {
        'asr.gs@calculate': {
            'calculator': {
                "name": "gpaw",
                "kpts": {"density": 2, "gamma": True},
                "xc": "PBE",
            },
        },
        'asr.bandstructure@calculate': {
            'npoints': 10,
            'emptybands': 5,
        },
        'asr.hse@calculate': {
            'kptdensity': 2,
            'emptybands': 5,
        },
        'asr.gw@gs': {
            'kptdensity': 2,
        },
        'asr.pdos@calculate': {
            'kptdensity': 2,
            'emptybands': 5,
        },
        'asr.piezoelectrictensor': {
            'calculator': {
                "name": "gpaw",
                "kpts": {"density": 2},
            },
        },
        'asr.formalpolarization': {
            'calculator': {
                "name": "gpaw",
                "kpts": {"density": 2},
            },
        },
    }

    write_json('params.json', params)
