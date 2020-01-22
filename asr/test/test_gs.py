import pytest
from pytest import approx


@pytest.mark.parametrize("efermi", [0, 0.25, 0.49])
@pytest.mark.parametrize("gap", [1, 2])
def test_gs_main(isolated_filesystem, mock_GPAW, gap, efermi):
    mock_GPAW.set_property(gap=gap, fermi_level=efermi)
    from asr.database.fromtree import main as fromtree
    from asr.gs import calculate, main

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
        }
    )

    results = main()
    if gap / 2 > efermi:
        assert results.get("gap") == approx(gap)
    else:
        assert results.get("gap") == approx(0)
    assert results.get("gaps_nosoc").get("efermi") == approx(efermi)
    assert results.get("efermi") == approx(efermi)
    fromtree()

    from asr.database import app as appmodule
    from pathlib import Path
    from asr.database.app import app, initialize_project, projects

    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    appmodule.tmpdir = tmpdir
    initialize_project("database.db")

    app.testing = True
    with app.test_client() as c:
        content = c.get(f"/database.db/").data.decode()
        assert "Fermi level" in content
        assert "Band gap" in content
        project = projects["database.db"]
        db = project["database"]
        uid_key = project["uid_key"]
        uids = []
        for row in db.select(include_data=False):
            uids.append(row.get(uid_key))
        for i, uid in enumerate(uids):
            url = f"/database.db/row/{uid}"
            content = c.get(url).data.decode()
            content = (
                content.replace(" ", "")
                .replace("<td>", "")
                .replace("</td>", "")
                .replace("\n", "")
            )
            assert f"Bandgap{gap:0.3f}eV" in content, content
            assert f"Fermilevel{efermi:0.3f}eV" in content, content
