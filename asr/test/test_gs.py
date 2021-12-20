import pytest
from pytest import approx
from .materials import Si, Fe
from pathlib import Path


@pytest.mark.ci
@pytest.mark.parallel
@pytest.mark.parametrize("gap", [0, 1])
@pytest.mark.parametrize("fermi_level", [0.5, 1.5])
def test_gs(asr_tmpdir_w_params, mockgpaw, mocker, get_webcontent,
            test_material, gap, fermi_level):
    import asr.c2db.relax
    from asr.core import read_json
    from asr.core.cache import get_cache
    from asr.c2db.gs import calculate, main
    from ase.parallel import world
    import gpaw
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    spy = mocker.spy(asr.c2db.relax, "set_initial_magnetic_moments")
    gpaw.GPAW._get_fermi_level.return_value = fermi_level
    gpaw.GPAW._get_band_gap.return_value = gap

    calculator = {'name': 'gpaw',
                  'kpts': {'density': 2, 'gamma': True},
                  'xc': 'PBE',
                  'mode': {'name': 'pw', 'ecut': 800}}
    calculateresult = calculate(test_material, calculator)
    record = main.get(
        atoms=test_material,
        calculator=calculator)
    results = record.result
    dependencies = record.dependencies
    cache = get_cache()
    dep_records = [
        cache.get(uid=dependency.uid, revision=dependency.revision)
        for dependency in dependencies
    ]
    dep_names = [dep_record.run_specification.name
                 for dep_record in dep_records]
    assert (set(dep_names)
            == set(['asr.c2db.gs:calculate', 'asr.c2db.magnetic_anisotropy:main']))
    gsfile = calculateresult.calculation.paths[0]
    assert Path(gsfile).is_file()
    gs = read_json(gsfile)
    gs['atoms'].has('initial_magmoms')
    if test_material.has('initial_magmoms'):
        spy.assert_not_called()
    else:
        spy.assert_called()

    assert len(list(
        Path('.asr/records').glob(
            'asr.c2db.magnetic_anisotropy*.json'))) == 1
    assert len(list(
        Path('.asr/records').glob('asr.c2db.gs:calculate*.json'))) == 1
    assert results.get("gaps_nosoc").get("efermi") == approx(fermi_level)
    assert results.get("efermi") == approx(fermi_level, abs=0.1)
    if gap >= fermi_level:
        assert results.get("gap") == approx(gap)
    else:
        assert results.get("gap") == approx(0)

    test_material.write('structure.json')
    if world.size == 1:
        from asr.structureinfo import main as structureinfo
        structureinfo(atoms=test_material)
        content = get_webcontent()
        resultgap = results.get("gap")

        # XXX When switching to webpanel2 we get three decimals rather
        # than two.  How in the name of all that is good and holy does
        # that happen here?
        assert f"{resultgap:0.3f}eV" in content, content
        assert "<td>Fermilevel</td>" in content, content
        assert "<td>Magneticstate</td><td>NM</td>" in \
            content, content
        assert 'Thispanel' in content, content
        assert "Electronicproperties" in content, content


@pytest.mark.ci
def test_gs_asr_cli_results_figures(asr_tmpdir_w_params, mockgpaw):
    from .materials import std_test_materials
    from asr.c2db.gs import main
    from asr.database import connect
    from asr.core.material import make_panel_figures
    from asr.core.datacontext import DataContext
    from asr.database.fromtree import collect_folders
    atoms = std_test_materials[0]
    atomsname = 'structure.json'
    atoms.write(atomsname)

    record = main.get(atoms=atoms)
    result = record.result

    dbname = 'database.db'
    # XXX Default values (None) cause function to fail.
    collect_folders(['.'], atomsname, dbname=dbname,
                    patterns=[], children_patterns=[])
    with connect(dbname) as conn:
        rows = list(conn.select())
    assert len(rows) == 1
    row = rows[0]

    context = DataContext(row, record, row.cache)
    panels = result.format_as('webpanel2', context)
    paths = make_panel_figures(context, panels, uid=record.uid[:10])

    assert len(paths) > 0
    for path in paths:
        assert path.is_file()

    # assert Path(f'{record.uid[:10]}-bz-with-gaps.png').is_file()


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
@pytest.mark.parametrize('atoms,calculator,results', [
    (Si,
     {
         "name": "gpaw",
         "kpts": {"density": 2, "gamma": True},
         "xc": "PBE",
         "mode": {"ecut": 300, "name": "pw"}
     },
     {'magstate': 'NM',
      'gap': pytest.approx(0.55, abs=0.01)}),
    (Fe,
     {
         "name": "gpaw",
         "kpts": {"density": 2, "gamma": True},
         "xc": "PBE",
         "mode": {"ecut": 300, "name": "pw"}
     },
     {'magstate': 'FM', 'gap': 0.0})
])
def test_gs_integration_gpaw(asr_tmpdir, atoms, calculator, results):
    """Check that the groundstates produced by GPAW are correct."""
    from asr.c2db.gs import main as groundstate
    from asr.c2db.magstate import main as magstate
    gsresults = groundstate(atoms=atoms, calculator=calculator)

    assert gsresults['gap'] == results['gap']

    magstateresults = magstate(atoms=atoms, calculator=calculator)
    assert magstateresults["magstate"] == results['magstate']
