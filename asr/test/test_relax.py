from asr.relax import BrokenSymmetryError
from pathlib import Path
import pytest
import numpy as np


@pytest.mark.ci
def test_relax_basic(asr_tmpdir_w_params, mockgpaw, test_material):
    """Test that the relaxation recipe actually produces a structure.json."""
    from asr.relax import main as relax
    from ase.io import write, read

    write('unrelaxed.json', test_material)
    relax(calculator={
        "name": "gpaw",
        "kpts": {"density": 2, "gamma": True},
    })
    read('structure.json')


@pytest.mark.ci
@pytest.mark.parametrize('initial_magmoms', ([0, 1]))
@pytest.mark.parametrize('set_magmoms', ([True, False]))
@pytest.mark.parametrize('final_magmoms', ([0, 1]))
def test_relax_magmoms(asr_tmpdir_w_params, mockgpaw, mocker, test_material,
                       initial_magmoms, set_magmoms, final_magmoms):
    """Test that the initial magnetic moments are correctly set."""
    import asr.relax
    from asr.relax import main
    from ase.io import read
    from gpaw import GPAW
    mocker.patch.object(GPAW, "_get_magmoms")
    spy = mocker.spy(asr.relax, "set_initial_magnetic_moments")
    GPAW._get_magmoms.return_value = (np.zeros((len(test_material), ), float)
                                      + final_magmoms)
    if set_magmoms:
        test_material.set_initial_magnetic_moments(
            [initial_magmoms] * len(test_material))

    test_material.write('unrelaxed.json')
    main()

    relaxed = read('structure.json')
    assert relaxed.has('initial_magmoms')

    if final_magmoms > 0.1:
        assert relaxed.get_initial_magnetic_moments().all()
    else:
        assert not relaxed.get_initial_magnetic_moments().any()

    # If user has set magnetic moments, make sure that we don't touch them
    if set_magmoms:
        spy.assert_not_called()
    else:
        spy.assert_called()


@pytest.mark.ci
@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_relax_emt(asr_tmpdir_w_params, name):
    from asr.relax import main as relax
    from ase.build import bulk

    unrelaxed = bulk(name)
    unrelaxed.write('unrelaxed.json')
    relax(calculator={'name': 'emt'})


@pytest.mark.ci
@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_relax_emt_fail_broken_symmetry(asr_tmpdir_w_params, name,
                                        monkeypatch, capsys):
    """Test that a broken symmetry raises an error."""
    from asr.relax import main as relax
    from ase.build import bulk
    from ase.calculators.emt import EMT

    unrelaxed = bulk(name)

    def get_stress(*args, **kwargs):
        return np.random.rand(3, 3)

    monkeypatch.setattr(EMT, 'get_stress', get_stress)
    unrelaxed.write('unrelaxed.json')
    with pytest.raises(BrokenSymmetryError) as excinfo:
        relax(calculator={'name': 'emt'}, enforce_symmetry=False)

    assert 'The symmetry was broken during the relaxation!' in str(excinfo.value)


@pytest.mark.ci
def test_relax_find_higher_symmetry(asr_tmpdir_w_params, monkeypatch, capsys):
    """Test that a structure is allowed to find a higher symmetry without failing."""
    from ase.build import bulk
    from asr.relax import main, SpgAtoms, myBFGS

    diamond = bulk('C')
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

    diamond.write('unrelaxed.json')

    monkeypatch.setattr(SpgAtoms, 'get_forces', get_forces)
    monkeypatch.setattr(SpgAtoms, 'get_stress', get_stress)
    monkeypatch.setattr(myBFGS, 'irun', irun)
    main(calculator={'name': 'emt'})

    captured = capsys.readouterr()
    assert "The spacegroup has changed during relaxation. " in captured.out


@pytest.mark.acceptance_test
def test_relax_fe_gpaw(asr_tmpdir):
    from asr.relax import main
    from ase.io import read
    from ase import Atoms
    a = 1.41973054
    magmom = 2.26739285
    Fe = Atoms('Fe',
               positions=[[0., 0., 0.]],
               cell=[[-a, a, a],
                     [a, -a, a],
                     [a, a, -a]],
               magmoms=[magmom],
               pbc=True)
    Fe.write('unrelaxed.json')
    main()
    relaxed = read('structure.json')
    magmoms = relaxed.get_initial_magnetic_moments()
    assert magmoms[0] == pytest.approx(magmom, abs=0.1)


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_relax_si_gpaw(asr_tmpdir_w_params):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax
    setupmaterial.cli(["-s", "Si2"])
    Path("materials.json").rename("unrelaxed.json")
    relaxargs = (
        "{'mode':{'ecut':200,'dedecut':'estimate',...},"
        "'kpts':{'density':1,'gamma':True},...}"
    )
    results = relax.cli(["--calculator", relaxargs])
    assert abs(results["c"] - 3.978) < 0.001


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_relax_bn_gpaw(asr_tmpdir_w_params):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax
    from asr.core import read_json

    setupmaterial.cli(["-s", "BN,natoms=2"])
    Path("materials.json").rename("unrelaxed.json")
    relaxargs = (
        "{'mode':{'ecut':300,'dedecut':'estimate',...},"
        "'kpts':{'density':2,'gamma':True},...}"
    )
    relax.cli(["--calculator", relaxargs])

    results = read_json("results-asr.relax.json")
    assert results["c"] > 5
