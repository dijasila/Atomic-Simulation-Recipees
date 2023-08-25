import pytest


@pytest.mark.ci
def test_spinspiral_calculate(asr_tmpdir, mockgpaw, test_material):
    """Test of spinspiral recipe."""
    from asr.spinspiral import calculate

    test_material.write('structure.json')
    calculate(q_c=[0.5, 0, 0], n=0, smooth=True)
    calculate(q_c=[0.5, 0, 0], n=0, smooth=True)  # test restart

    calculate(q_c=[0.5, 0, 0], n=1, smooth=False)


@pytest.mark.ci
@pytest.mark.parametrize("path_data", [(None, 0), ('G', 0), ('111', 4)])
def test_spinspiral_main(asr_tmpdir, test_material, mockgpaw, get_webcontent,
                         mocker, path_data):
    from asr.spinspiral import main
    from asr.core import ASRResult

    test_material.write('structure.json')

    def calculate(q_c=[0, 0, 0], n=0, params=dict(), smooth=True):
        return ASRResult.fromdata(en=0,
                                  q=q_c,
                                  ml=[[0, 0, 0]],
                                  mT=[0, 0, 0],
                                  gap=0)

    mocker.patch('asr.spinspiral.calculate', create=True, new=calculate)

    magmoms = [[1, 0, 0]] * len(test_material)
    params = {
        "mode": {
            "mode": "pw",
            "ecut": 300
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

    q_path, npoints = path_data
    main(q_path=q_path,
         npoints=npoints,
         params=params,
         smooth=True,
         clean_up=True,
         eps=0.2)
    # from os import listdir
    # print(listdir('.'))
    # print(test_material.get_positions())


@pytest.mark.ci
def test_spinspiral_integration(asr_tmpdir, test_material, mockgpaw,
                                get_webcontent, mocker):
    from asr.spinspiral import main
    test_material.write('structure.json')

    magmoms = [[1, 0, 0]] * len(test_material)
    params = {
        "mode": {
            "mode": "pw",
            "ecut": 300
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

    main(q_path=None,
         npoints=0,
         params=params,
         smooth=True,
         clean_up=True,
         eps=0.2)
