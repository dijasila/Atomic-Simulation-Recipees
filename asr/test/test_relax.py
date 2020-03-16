from .conftest import test_materials
from asr.relax import BrokenSymmetryError
from pathlib import Path
import pytest
import numpy as np


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
                                        monkeypatch, capsys):
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
    with pytest.raises(BrokenSymmetryError) as excinfo:
        relax(calculator={'name': 'emt'}, enforce_symmetry=False)

    assert 'The symmetry was broken during the relaxation!' in str(excinfo.value)


generators = [
    [[1, 0, 0],
     [0, -1, 0],
     [0, 0, 1]],
    [[0, 1, 0],
     [1, 0, 0],
     [0, 0, 1]]
]
fractrans_generators = [[0, 0, 0], [0, 0, 0.5]]
@pytest.mark.ci
@pytest.mark.unittest
@pytest.mark.parametrize("generator", generators)
@pytest.mark.parametrize("fractrans_generator", fractrans_generators)
def test_relax_find_common_symmetries(generator, fractrans_generator):
    from asr.relax import find_common_symmetries

    gen = np.zeros((4, 4), float)
    gen[0, 0] = 1
    gen[1:, 0] = fractrans_generator
    gen[1:, 1:] = generator

    def generate_symmetry_list(inputsym):
        # Only works for cyclic group of order < 10
        allsyms = []
        for n in range(1, 10):
            sym = np.linalg.matrix_power(inputsym, n)
            allsyms.append(sym.copy())
            sym[1:, 0] -= np.round(sym[1:, 0])
            if np.allclose(sym, np.eye(4)):
                break
        return allsyms

    syms = generate_symmetry_list(gen)
    print(syms)
    nsyms = len(syms)
    print(nsyms)

    common_symmetries, index1, index2 = find_common_symmetries(syms, syms)
    assert not index1
    assert not index2
    assert len(common_symmetries) == nsyms

    common_symmetries, index1, index2 = find_common_symmetries(syms, [np.eye(4)])
    assert len(common_symmetries) == 1
    assert nsyms - 1 not in index1
    assert len(index1) == nsyms - 1
    assert not index2

    common_symmetries, index1, index2 = find_common_symmetries([np.eye(4)], syms)
    assert len(common_symmetries) == 1
    assert nsyms - 1 not in index2
    assert len(index2) == nsyms - 1
    assert not index1


@pytest.mark.ci
def test_relax_find_higher_symmetry(separate_folder, monkeypatch, capsys):
    """Test that a structure is allowed to find a higher symmetry without failing."""
    from ase.build import bulk
    from asr.relax import main, SpgAtoms, myBFGS
    import numpy as np

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
