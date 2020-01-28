import pytest
from pytest import approx
from .conftest import test_materials
from hypothesis import given
from hypothesis.strategies import floats


def get_webcontent(name='database.db'):
    from asr.database.fromtree import main as fromtree
    fromtree()

    from asr.database import app as appmodule
    from pathlib import Path
    from asr.database.app import app, initialize_project, projects

    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    appmodule.tmpdir = tmpdir
    initialize_project(name)

    app.testing = True
    with app.test_client() as c:
        content = c.get(f"/database.db/").data.decode()
        assert "Fermi level" in content
        assert "Band gap" in content
        project = projects["database.db"]
        db = project["database"]
        uid_key = project["uid_key"]
        row = db.get(id=1)
        uid = row.get(uid_key)
        url = f"/database.db/row/{uid}"
        content = c.get(url).data.decode()
        content = (
            content.replace(" ", "")
            .replace("<td>", "")
            .replace("</td>", "")
            .replace("\n", "")
        )
    return content


run_no = 0
@pytest.mark.parametrize("atoms", test_materials)
@given(gap=floats(min_value=0, max_value=10),
       fermi_level=floats(min_value=0, max_value=1))
def test_gs_main(separate_folder, usemocks, atoms, gap, fermi_level):
    global run_no
    run_no += 1
    with separate_folder(path=f'run-{run_no}'):
        from gpaw import GPAW as GPAWMOCK
        GPAWMOCK.set_property(gap=gap, fermi_level=fermi_level)

        from asr.gs import calculate, main
        from ase.io import write

        write('structure.json', atoms)
        calculate(
            calculator={
                "name": "gpaw",
                "mode": {"name": "pw", "ecut": 800},
                "xc": "PBE",
                "basis": "dzp",
                "kpts": {"density": 2, "gamma": True},
                "occupations": {"name": "fermi-dirac", "width": 0.05},
                "convergence": {"bands": "CBM+3.0"},
                "nbands": "200%",
                "txt": "gs.txt",
                "charge": 0,
            },
            # skip_deps=True
        )

        results = main()
        if gap > fermi_level:
            assert results.get("gap") == approx(gap)
        else:
            assert results.get("gap") == approx(0)
        assert results.get("efermi") == approx(fermi_level)
        assert results.get("gaps_nosoc").get("efermi") == approx(fermi_level)

        content = get_webcontent('database.db')
        assert f"Bandgap{gap:0.3f}eV" in content, content
        assert f"Fermilevel{fermi_level:0.3f}eV" in content, content
