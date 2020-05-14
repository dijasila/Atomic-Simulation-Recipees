import pytest
from pytest import approx
from .materials import Si, Fe


@pytest.mark.ci
@pytest.mark.parallel
@pytest.mark.parametrize("gap", [0, 1])
@pytest.mark.parametrize("fermi_level", [0.5, 1.5])
def test_gs(asr_tmpdir_w_params, mockgpaw, mocker, get_webcontent,
            test_material, gap, fermi_level):
    import asr.relax
    from asr.core import read_json
    from asr.gs import calculate, main
    from ase.io import write
    import gpaw
    import gpaw.occupations
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    spy = mocker.spy(asr.relax, "set_initial_magnetic_moments")
    gpaw.GPAW._get_fermi_level.return_value = fermi_level
    gpaw.GPAW._get_band_gap.return_value = gap

    write('structure.json', test_material)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )

    results = main()
    gs = read_json('gs.gpw')
    gs['atoms'].has('initial_magmoms')
    if test_material.has('initial_magmoms'):
        spy.assert_not_called()
    else:
        spy.assert_called()

    assert results.get("gaps_nosoc").get("efermi") == approx(fermi_level)
    assert results.get("efermi") == approx(fermi_level)
    if gap >= fermi_level:
        assert results.get("gap") == approx(gap)
    else:
        assert results.get("gap") == approx(0)

    content = get_webcontent()
    resultgap = results.get("gap")
    assert f"<td>Bandgap</td><td>{resultgap:0.2f}eV</td>" in content, content
    assert f"<td>Fermilevel</td><td>{fermi_level:0.3f}eV</td>" in \
        content, content
    assert "<td>Magneticstate</td><td>NM</td>" in \
        content, content


@pytest.mark.ci
def test_gs_asr_cli_results_figures(asr_tmpdir_w_params, mockgpaw):
    from .materials import std_test_materials
    from pathlib import Path
    from asr.gs import main
    from asr.core.material import (get_material_from_folder,
                                   get_webpanels_from_material,
                                   make_panel_figures)
    atoms = std_test_materials[0]
    atoms.write('structure.json')

    main()
    material = get_material_from_folder()
    panel = get_webpanels_from_material(material, main)
    make_panel_figures(material, panel)
    assert Path('bz-with-gaps.png').is_file()


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
@pytest.mark.parametrize('atoms,parameters,magstate', [
    (Si,
     {
         'asr.gs@calculate': {
             'calculator': {
                 "name": "gpaw",
                 "kpts": {"density": 2, "gamma": True},
                 "xc": "PBE",
                 "mode": {"ecut": 300, "name": "pw"}
             },
         }
     },
     'NM'),
    (Fe,
     {
         'asr.gs@calculate': {
             'calculator': {
                 "name": "gpaw",
                 "kpts": {"density": 2, "gamma": True},
                 "xc": "PBE",
                 "mode": {"ecut": 300, "name": "pw"}
             },
         }
     },
     'FM')
])
def test_gs_integration_gpaw(asr_tmpdir, atoms, parameters, magstate):
    """Check that the groundstates produced by GPAW are correct."""
    from asr.core import read_json
    from asr.gs import main as groundstate
    from asr.setup.params import main as setupparams
    atoms.write('structure.json')
    setupparams(parameters)
    groundstate()
    magstateresults = read_json('results-asr.magstate.json')
    assert magstateresults["magstate"] == magstate
