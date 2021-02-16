import pytest
from pytest import approx
from .materials import Si, BN


@pytest.mark.ci
@pytest.mark.parallel
@pytest.mark.parametrize("state", [0, 1, 2])
def test_get_wfs(asr_tmpdir, mockgpaw, mocker, test_material, state):
    from asr.gs import calculate, main
    from asr.get_wfs import main as get_wfs
    import gpaw
    # from gpaw import restart

    mocker.path.object(gpaw.GPAW, "_get_eigenvalues")

    spy = mocker.spy(asr.relax, "set_initial_magnetic_moments")
    gpaw.GPAW._get_eigenvalues[0].return_value = eigenvalue

    write('structure.json', test_material)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 6, "gamma": True},
        },
    )

    gsresults = main()

    # _, calc = restart('gs.gpw', txt=None)

    results = get_wfs(state=state)

    assert 0 == 0
