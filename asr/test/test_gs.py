import pytest
from pytest import approx
from .conftest import test_materials


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


@pytest.mark.parametrize("atoms", test_materials)
@pytest.mark.parametrize("efermi", [0.5])
@pytest.mark.parametrize("gap", [0, 1.0])
def test_gs_main(isolated_filesystem, mock_gpaw, gap, efermi, atoms):
    mock_gpaw.set_property(gap=gap, fermi_level=efermi)
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
        skip_deps=True
    )

    results = main()
    if gap > efermi:
        assert results.get("gap") == approx(gap)
    else:
        assert results.get("gap") == approx(0)
    assert results.get("gaps_nosoc").get("efermi") == approx(efermi)
    assert results.get("efermi") == approx(efermi)

    content = get_webcontent('database.db')
    assert f"Bandgap{gap:0.3f}eV" in content, content
    assert f"Fermilevel{efermi:0.3f}eV" in content, content
