import pytest
from pytest import approx
from .conftest import test_materials, get_webcontent, freeelectroneigenvalues


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
@pytest.mark.parametrize("gap", [0, 1])
@pytest.mark.parametrize("fermi_level", [0.5, 1.5])
def test_gs(separate_folder, mockcalculator, mocker, atoms, gap, fermi_level):
    from asr.gs import calculate, main
    from ase.io import write
    from ase.units import Ha
    import gpaw
    import gpaw.occupations
    get_eigenvalues = freeelectroneigenvalues(atoms, gap=gap)

    mocker.patch.object(gpaw.GPAW, "get_eigenvalues", new=get_eigenvalues)
    mocker.patch.object(gpaw.GPAW, "get_fermi_level")
    mocker.patch("gpaw.occupations.occupation_numbers")
    gpaw.GPAW.get_fermi_level.return_value = fermi_level
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
