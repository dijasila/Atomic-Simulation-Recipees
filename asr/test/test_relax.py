from .conftest import test_materials
from asr.relax import BrokenSymmetryError
from pathlib import Path
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_relax(separate_folder, mockgpaw, atoms):
    from asr.relax import main as relax
    from ase.io import write

    write('unrelaxed.json', atoms)
    relax(calculator={
        "name": "gpaw",
        "kpts": {"density": 2, "gamma": True},
    })


@pytest.mark.ci
@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_relax_emt(separate_folder, name):
    from asr.relax import main as relax
    from ase.build import bulk

    unrelaxed = bulk(name)
    unrelaxed.write('unrelaxed.json')
    relax(calculator={'name': 'emt'})


@pytest.mark.ci
@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_relax_emt_fail_broken_symmetry(separate_folder, name,
                                        monkeypatch):
    """Test that a broken symmetry raises an error."""
    from asr.relax import main as relax
    from ase.build import bulk
    import numpy as np
    from ase.calculators.emt import EMT

    unrelaxed = bulk(name)

    def get_stress(*args, **kwargs):
        return np.random.rand(3, 3)

    monkeypatch.setattr(EMT, 'get_stress', get_stress)
    unrelaxed.write('unrelaxed.json')
    with pytest.raises(BrokenSymmetryError):
        relax(calculator={'name': 'emt'}, enforce_symmetry=False)


@pytest.mark.ci
def test_relax_find_higher_symmetry(separate_folder, monkeypatch):
    """Test that a structure is allowed to find a higher symmetry without failing."""
    from ase.build import bulk
    from ase.atoms import Atoms
    from asr.relax import main
    from ase.calculators.emt import EMT
    import numpy as np

    diamond = bulk('C')
    sposoriginal_ac = diamond.get_scaled_positions()
    spos_ac = diamond.get_scaled_positions()
    spos_ac[1][2] += 0.1
    diamond.set_scaled_positions(spos_ac)

    def get_stress(*args, **kwargs):
        return np.zeros((3, 3), float)

    monkeypatch.setattr(EMT, 'get_stress', get_stress)

    def set_positions(self, *args, **kwargs):
        return self.set_scaled_positions(sposoriginal_ac)

    monkeypatch.setattr(Atoms, 'set_positions', set_positions)

    diamond.write('unrelaxed.json')
    main(calculator={'name': 'emt'})


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_relax_si_gpaw(separate_folder):
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
def test_relax_bn_gpaw(separate_folder):
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
