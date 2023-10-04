from ase.parallel import world
import pytest


def get_calculator_default(qspiral=None, magmoms=None):
    calculator = {
        "mode": {
            "name": "pw",
            "ecut": 150,
        },
        'soc': False,
        "kpts": {
            "density": 0.5,
            "gamma": True
        },
        'txt': 'gsq0b0.txt'
    }
    if magmoms is not None:
        calculator["magmoms"] = magmoms
    if qspiral is not None:
        calculator["mode"]["qspiral"] = qspiral
    return calculator


@pytest.mark.ci
@pytest.mark.skipif(world.size > 1, reason='Job submission is serial')
def test_unconverged_skip(asr_tmpdir, test_material):
    """Test of skipping non-converged calcs upon resubmission."""
    from asr.spinspiral import cannot_converge
    with open('gsq0b0.txt', 'w'):
        pass

    # A single .txt file can be caused by timeout, so convergence isinconclusive
    assert not cannot_converge(qidx=0, bidx=0)

    with open('gsq1b0.txt', 'w'), open('gsq2b0.txt', 'w'):
        pass

    # Two subsequent .txt files without a .gpw of the first means convergence
    # of the former has failed
    assert cannot_converge(qidx=1, bidx=0)
    assert not cannot_converge(qidx=2, bidx=0)


@pytest.mark.ci
@pytest.mark.parametrize("path_data", [(None, 0), ('G', 0)])
def test_spinspiral_main(asr_tmpdir, test_material, mocker, path_data):
    """Test of spinspiral recipe."""
    from asr.spinspiral import main

    test_material.write('structure.json')

    def spinspiral(calculator={'qspiral': [0.5, 0, 0], 'txt': 'gsq0.txt'}):
        return {'energy': 1, 'totmom_v': [1, 0, 0],
                'magmom_av': [[1, 0, 0]], 'gap': 1}

    mocker.patch('asr.spinspiral.spinspiral', create=True, new=spinspiral)

    magmoms = [[1, 0, 0]] * len(test_material)
    qspiral = [0.5, 0, 0]
    calculator = get_calculator_default(qspiral=qspiral, magmoms=magmoms)

    q_path, qpts = path_data
    main(calculator=calculator,
         q_path=q_path,
         qpts=qpts,
         rotation_model='q.a',
         clean_up=True,
         eps=0.2)


@pytest.mark.ci
@pytest.mark.parallel
def test_hchain_integration(asr_tmpdir, get_webcontent):
    from ase.parallel import world
    from ase import Atoms
    from asr.spinspiral import main
    from asr.collect_spiral import main as collect
    from asr.core import read_json
    from numpy import array
    atoms = Atoms('H', cell=[3, 3, 3], pbc=[False, True, False])
    atoms.center()
    atoms.write('structure.json')
    magmoms = array([[1, 0, 0]])
    calculator = get_calculator_default(magmoms=magmoms)

    main(calculator=calculator, qpts=3)
    collect()

    if world.size == 1:
        content = get_webcontent()
        print(content)
        assert '<td>Q<sub>min</sub></td>' in content, content
        assert '<td>Bandgap(Q<sub>min</sub>)(eV)</td>' in content, content
        assert '<td>Spiralbandwidth(meV)</td>' in content, content

        result = read_json('results-asr.collect_spiral.json')
        # Non-magnetic hchain is metallic.
        assert (result['gaps'] > 0).all(), result['gaps']

        # This ensure correct energy difference between q=0 and q=X
        assert result['bandwidth'] == pytest.approx(81.2068999814044, abs=1e-6)


@pytest.mark.ci
def test_initial_magmoms(test_material):
    """Test of magnetic moment initialization with differen models."""
    from asr.utils.spinspiral import extract_magmoms, rotate_magmoms
    magmoms = [[1, 0, 0]] * len(test_material)
    qspiral = [0.5, 0, 0]
    calculator = get_calculator_default(qspiral=qspiral, magmoms=magmoms)

    init_magmoms = extract_magmoms(test_material, calculator)
    rotate_magmoms(test_material, init_magmoms, qspiral, 'q.a')
    rotate_magmoms(test_material, init_magmoms, qspiral, 'tan')
