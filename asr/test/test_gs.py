import pytest
from pytest import approx
from .materials import Si, Fe
from pathlib import Path


@pytest.mark.ci
@pytest.mark.parallel
@pytest.mark.parametrize("gap", [0, 1])
@pytest.mark.parametrize("fermi_level", [0.5, 1.5])
def test_gs(asr_tmpdir_w_params,
            repo, mockgpaw, mocker, get_webcontent,
            test_material, gap, fermi_level):
    import asr.c2db.relax
    from asr.core import read_json
    from asr.c2db.gs import workflow
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

    dct = repo.run_workflow_blocking(
        workflow, atoms=test_material, calculator=calculator)

    calculateresult = dct['gs'].value().output
    post = dct['postprocess'].value().output

    gsfile = calculateresult.calculation.paths[0]
    assert Path(gsfile).is_file()
    gs = read_json(gsfile)
    gs['atoms'].has('initial_magmoms')
    if test_material.has('initial_magmoms'):
        spy.assert_not_called()
    else:
        spy.assert_called()

    assert post.get("gaps_nosoc").get("efermi") == approx(fermi_level)
    assert post.get("efermi") == approx(fermi_level, abs=0.1)

    if gap >= fermi_level:
        assert post.get("gap") == approx(gap)
    else:
        assert post.get("gap") == approx(0)

    test_material.write('structure.json')


@pytest.mark.xfail
def test_gs_structureinfo():
    # This snippet was part of the preceding test but we cannot call it until
    # htw caches can be collected to a database
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
@pytest.mark.parametrize('atoms,refs', [
    (Si,
     {'magstate': 'NM',
      'gap': pytest.approx(0.55, abs=0.01)}),
    (Fe, {'magstate': 'FM', 'gap': 0.0})
])
def test_gs_integration_gpaw(repo, atoms, refs):
    """Check that the groundstates produced by GPAW are correct."""
    from asr.c2db.gs import workflow
    # from htwutil.runner import Runner

    calculator = {
        'txt': 'gpaw.txt',
        'name': 'gpaw',
        'xc': 'PBE',
        'kpts': {'density': 2.0, 'gamma': True},
        'mode': {'ecut': 200, 'name': 'pw'},
    }

    dct = repo.run_workflow_blocking(workflow,
                                     atoms=atoms, calculator=calculator)

    outputs = {name: future.value().output
               for name, future in dct.items()}

    assert outputs['postprocess']['gap'] == refs['gap']
    assert outputs['magstate']['magstate'] == refs['magstate']
