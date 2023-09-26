import pytest


@pytest.mark.ci
def test_spinspiral_calculate(asr_tmpdir, mockgpaw, test_material):
    """Test of spinspiral recipe."""
    from asr.spinspiral import spinspiral
    from numpy import array
    test_material.write('structure.json')
    spinspiral(calculator={"mode":
               {"name": "pw", "ecut": 300, 'qspiral': [0.5, 0, 0]},
               'txt': 'gsq0b0.txt'})
    spinspiral(calculator={"mode":
               {"name": "pw", "ecut": 300, 'qspiral': [0.5, 0, 0]},
               'txt': 'gsq0b0.txt'})  # test restart

    res = spinspiral(calculator={"mode":
                     {"name": "pw", "ecut": 300, 'qspiral': [0.5, 0, 0]},
                     'txt': 'gsq1b0.txt'})

    assert (res['totmom_v'] == array([1., 1., 1.])).all()
    assert res['energy'] == 0.0


@pytest.mark.ci
def test_unconverged_skip(asr_tmpdir, mockgpaw, test_material):
    """Test of spinspiral recipe."""
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
def test_spinspiral_main(asr_tmpdir, test_material, mockgpaw, get_webcontent,
                         mocker, path_data):
    from asr.spinspiral import main

    test_material.write('structure.json')

    def spinspiral(calculator={'qspiral': [0.5, 0, 0], 'txt': 'gsq0.txt'}):
        return {'energy': 1, 'totmom_v': [1, 0, 0],
                'magmom_av': [[1, 0, 0]], 'gap': 1}

    mocker.patch('asr.spinspiral.spinspiral', create=True, new=spinspiral)

    magmoms = [[1, 0, 0]] * len(test_material)
    calculator = {
        "mode": {
            "name": "pw",
            "ecut": 300,
            'qspiral': [0.5, 0, 0]
        },
        "experimental": {
            'magmoms': magmoms,
            'soc': False
        },
        "kpts": {
            "density": 6,
            "gamma": True
        }
    }

    q_path, qpts = path_data
    main(calculator=calculator,
         q_path=q_path,
         qpts=qpts,
         rotation_model='q.a',
         clean_up=True,
         eps=0.2)


@pytest.mark.ci
def test_spinspiral_integration(asr_tmpdir, mocker,
                                test_material, mockgpaw, get_webcontent):
    from ase.parallel import world
    from asr.spinspiral import main
    from asr.collect_spiral import main as collect
    test_material.write('structure.json')

    # Spin spiral plotting uses E=0 to determine failed calculations
    mocker.patch('gpaw.GPAW._get_potential_energy', return_value=1.0)
    magmoms = [[1, 0, 0]] * len(test_material)
    calculator = {
        "mode": {
            "name": "pw",
            "ecut": 300,
        },
        "experimental": {
            'magmoms': magmoms,
            'soc': False
        },
        "kpts": {
            "density": 6,
            "gamma": True
        }
    }

    main(calculator=calculator, qpts=3)
    collect()

    if world.size == 1:
        content = get_webcontent()
        assert '<td>Q<sub>min</sub></td><td>[0.0.0.]</td>' in content
        assert '<td>Bandgap(Q<sub>min</sub>)(eV)</td><td>0.0</td>' in content
        assert '<td>Spiralbandwidth(meV)</td><td>0.0</td>' in content


def test_initial_magmoms(test_material):
    from asr.utils.spinspiral import extract_magmoms, rotate_magmoms
    magmoms = [[1, 0, 0]] * len(test_material)
    q_c = [0.5, 0, 0]
    calculator = {
        "mode": {
            "name": "pw",
            "ecut": 300,
            'qspiral': q_c
        },
        "experimental": {
            'magmoms': magmoms,
            'soc': False
        },
        "kpts": {
            "density": 6,
            "gamma": True
        }
    }

    init_magmoms = extract_magmoms(test_material, calculator)
    rotate_magmoms(test_material, init_magmoms, q_c, 'q.a')
    rotate_magmoms(test_material, init_magmoms, q_c, 'tan')
