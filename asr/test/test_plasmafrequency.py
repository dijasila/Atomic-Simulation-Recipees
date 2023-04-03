import pytest
from asr.c2db.plasmafrequency import PlasmaFrequencyWorkflow


def workflow(rn, atoms, calculator):
    scf = rn.task(
        'asr.c2db.gs.calculate',
        name='gscalculate',
        atoms=atoms,
        calculator=calculator)

    return PlasmaFrequencyWorkflow(rn, scf.output)


@pytest.mark.ci
def test_plasmafrequency(repo, get_webcontent, mockgpaw,
                         test_material, fast_calc):
    """Test of the plasma freuquency recipe."""
    if sum(test_material.pbc) != 2:
        pytest.skip("Plasma frequency is only implemented for 2D atm.")

    wf = repo.run_workflow_blocking(
        workflow,
        atoms=test_material,
        calculator=fast_calc)

    with repo:
        xfreq = wf.postprocess.value().output['plasmafrequency_x']
        assert xfreq == pytest.approx(0)

    # test_material.write('structure.json')
    # content = get_webcontent()
    # assert "plasmafrequency" in content
