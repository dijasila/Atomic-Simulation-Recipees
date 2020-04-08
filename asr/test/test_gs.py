import pytest
from pytest import approx
from .conftest import test_materials, get_webcontent


@pytest.mark.ci
@pytest.mark.parallel
@pytest.mark.parametrize("atoms", test_materials)
@pytest.mark.parametrize("gap", [0, 1])
@pytest.mark.parametrize("fermi_level", [0.5, 1.5])
def test_gs(separate_folder, mockgpaw, mocker, atoms, gap, fermi_level):
    from asr.gs import calculate, main
    from ase.io import write
    from ase.units import Ha
    import gpaw
    import gpaw.occupations
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    mocker.patch("gpaw.occupations.occupation_numbers")
    gpaw.GPAW._get_fermi_level.return_value = fermi_level
    gpaw.GPAW._get_band_gap.return_value = gap
    gpaw.occupations.occupation_numbers.return_value = [0,
                                                        fermi_level / Ha,
                                                        0,
                                                        0]

    write('structure.json', atoms)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )

    results = main()
    assert results.get("gaps_nosoc").get("efermi") == approx(fermi_level)
    assert results.get("efermi") == approx(fermi_level)
    if gap >= fermi_level:
        assert results.get("gap") == approx(gap)
    else:
        assert results.get("gap") == approx(0)

    content = get_webcontent('database.db')
    resultgap = results.get("gap")
    assert f"<td>Bandgap</td><td>{resultgap:0.2f}eV</td>" in content, content
    assert f"<td>Fermilevel</td><td>{fermi_level:0.3f}eV</td>" in \
        content, content


@pytest.mark.ci
def test_gs_asr_cli_results_figures(separate_folder, mockgpaw):
    from pathlib import Path
    from asr.gs import main
    from asr.core.material import (get_material_from_folder,
                                   get_webpanels_from_material,
                                   make_panel_figures)
    atoms = test_materials[0]
    atoms.write('structure.json')

    main()
    material = get_material_from_folder()
    panel = get_webpanels_from_material(material, main)
    make_panel_figures(material, panel)
    assert Path('bz-with-gaps.png').is_file()
