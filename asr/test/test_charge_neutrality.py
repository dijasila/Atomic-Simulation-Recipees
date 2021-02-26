import pytest
from .materials import BN


@pytest.mark.ci
def test_p_type(asr_tmpdir):
    from ase.io import write
    from asr.gs import calculate, main
    from asr.charge_neutrality import main as charge_neutrality

    write('structure.json', BN)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )
    main()

    ptype_dict = {'testdef': [(0.2, 0), (0.09, -1), (1., 1)]}
    result = charge_neutrality(defects=ptype_dict)

    gap = result['gap']
    ef = result['efermi_sc']

    assert ef < (gap * 0.25)


@pytest.mark.ci
def test_n_type(asr_tmpdir):
    from ase.io import write
    from asr.gs import calculate, main
    from asr.charge_neutrality import main as charge_neutrality

    write('structure.json', BN)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )
    main()

    ntype_dict = {'testdef': [(0.2, 0), (4.09, -1), (-3.2, 1)]}
    result = charge_neutrality(defects=ntype_dict)

    gap = result['gap']
    ef = result['efermi_sc']

    assert ef > (gap * 0.75)


@pytest.mark.ci
def test_undopable(asr_tmpdir):
    from ase.io import write
    from asr.gs import calculate, main
    from asr.charge_neutrality import main as charge_neutrality

    write('structure.json', BN)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )
    main()

    undop_dict = {'testdef': [(5.2, 0), (6.09, -2), (4.2, 3)]}
    result = charge_neutrality(defects=undop_dict)

    gap = result['gap']
    ef = result['efermi_sc']

    assert ef > (gap * 0.25)
    assert ef < (gap * 0.75)


@pytest.mark.ci
def test_multiple(asr_tmpdir):
    from ase.io import write
    from asr.gs import calculate, main
    from asr.charge_neutrality import main as charge_neutrality

    write('structure.json', BN)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )
    main()

    mult_dict = {'testdef1': [(5.2, 0), (6.09, -2), (4.2, 3)],
                 'testdef2': [(5.2, 0), (6.09, -2), (2.2, 2)],
                 'testdef3': [(1.2, 0), (0.0, -1), (0.2, 1)],
                 'testdef4': [(1.2, 0), (5.09, -1), (-4.5, 1)]}
    result = charge_neutrality(defects=mult_dict)

    gap = result['gap']
    ef = result['efermi_sc']

    assert ef > (gap * 0.25)
    assert ef < (gap * 0.75)
