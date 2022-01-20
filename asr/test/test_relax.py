from asr.c2db.relax import BrokenSymmetryError
import pytest
import numpy as np


@pytest.mark.ci
def test_relax_basic(asr_tmpdir_w_params, mockgpaw, test_material):
    """Test that the relaxation recipe actually produces a structure.json."""
    from asr.c2db.relax import main as relax
    relax(test_material,
          calculator={
              "name": "gpaw",
              "kpts": {"density": 2, "gamma": True},
          })


@pytest.mark.ci
@pytest.mark.parametrize('initial_magmoms', ([0, 1]))
@pytest.mark.parametrize('set_magmoms', ([True, False]))
@pytest.mark.parametrize('final_magmoms', ([0, 1]))
def test_relax_magmoms(asr_tmpdir_w_params, mockgpaw, mocker, test_material,
                       initial_magmoms, set_magmoms, final_magmoms):
    """Test that the initial magnetic moments are correctly set."""
    import asr.c2db.relax
    from asr.c2db.relax import main
    from gpaw import GPAW

    mocker.patch.object(GPAW, "_get_magmoms")
    spy = mocker.spy(asr.c2db.relax, "set_initial_magnetic_moments")
    GPAW._get_magmoms.return_value = (np.zeros((len(test_material), ), float)
                                      + final_magmoms)
    if set_magmoms:
        test_material.set_initial_magnetic_moments(
            [initial_magmoms] * len(test_material))

    if not test_material.has('initial_magmoms'):
        attempt_spinpol = True
    elif any(test_material.get_initial_magnetic_moments()):
        attempt_spinpol = True
    else:
        attempt_spinpol = False

    test_material.write('unrelaxed.json')
    record = main.cli([])
    relaxed = record.result.atoms

    assert relaxed.has('initial_magmoms')

    if final_magmoms > 0.1 and attempt_spinpol:
        assert all(record.result.magmoms == 1)
    else:
        assert not relaxed.get_initial_magnetic_moments().any()

    # If user has set magnetic moments, make sure that we don't touch them
    if test_material.has('initial_magmoms'):
        spy.assert_not_called()
    else:
        spy.assert_called()


@pytest.mark.ci
@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_relax_emt(asr_tmpdir_w_params, name):
    from asr.c2db.relax import main as relax
    from ase.build import bulk

    unrelaxed = bulk(name)
    unrelaxed.set_initial_magnetic_moments([0.0] * len(unrelaxed))
    relax(unrelaxed, calculator={'name': 'emt'})


@pytest.mark.ci
@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_relax_emt_fail_broken_symmetry(asr_tmpdir_w_params, name,
                                        monkeypatch, capsys):
    """Test that a broken symmetry raises an error."""
    from asr.c2db.relax import main as relax
    from ase.build import bulk
    from ase.calculators.emt import EMT

    unrelaxed = bulk(name)

    rng = np.random.RandomState(1234)

    def get_stress(*args, **kwargs):
        return rng.rand(3, 3)

    monkeypatch.setattr(EMT, 'get_stress', get_stress)
    with pytest.raises(BrokenSymmetryError) as excinfo:
        relax(unrelaxed, calculator={'name': 'emt'}, enforce_symmetry=False)

    assert 'The symmetry was broken during the relaxation!' in str(excinfo.value)


@pytest.mark.ci
def test_relax_find_higher_symmetry(asr_tmpdir_w_params, monkeypatch, capsys):
    """Test that a structure is allowed to find a higher symmetry without failing."""
    from ase.build import bulk
    from asr.c2db.relax import main, SpgAtoms, myBFGS

    diamond = bulk('C')
    diamond.set_initial_magnetic_moments([0.0] * len(diamond))
    natoms = len(diamond)
    sposoriginal_ac = diamond.get_scaled_positions()
    spos_ac = diamond.get_scaled_positions()
    spos_ac[1][2] += 0.1
    diamond.set_scaled_positions(spos_ac)

    def get_stress(*args, **kwargs):
        return np.zeros((6,), float)

    def get_forces(*args, **kwargs):
        return 1 + np.zeros((natoms, 3), float)

    def irun(self, *args, **kwargs):
        yield False
        self.atoms.atoms.set_scaled_positions(sposoriginal_ac)
        yield False

    monkeypatch.setattr(SpgAtoms, 'get_forces', get_forces)
    monkeypatch.setattr(SpgAtoms, 'get_stress', get_stress)
    monkeypatch.setattr(myBFGS, 'irun', irun)
    main(diamond, calculator={'name': 'emt'})

    captured = capsys.readouterr()
    assert "The spacegroup has changed during relaxation. " in captured.out


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_relax_si_gpaw(asr_tmpdir):
    from asr.c2db.relax import main as relax
    from .materials import Si
    calculator = {}
    calculator.update(relax.defaults.calculator)
    calculator['mode'] = {
        'ecut': 200,
        'dedecut': 'estimate',
        'name': 'pw',
    }
    calculator['kpts'] = {'density': 1, 'gamma': True}
    results = relax(
        atoms=Si.copy(),
        calculator=calculator,
    )
    assert abs(results["c"] - 3.978) < 0.1


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_relax_bn_gpaw(asr_tmpdir):
    from .materials import BN
    from asr.c2db.relax import main as relax

    calculator = {}
    calculator.update(relax.defaults.calculator)
    calculator['mode'] = {
        'ecut': 300,
        'dedecut': 'estimate',
        'name': 'pw',
    }
    calculator['kpts'] = {'density': 2, 'gamma': True}
    results = relax(atoms=BN.copy(), calculator=calculator)

    assert results["c"] > 5
