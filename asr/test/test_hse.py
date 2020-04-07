import pytest

from .conftest import test_materials, get_webcontent


@pytest.mark.ci
@pytest.mark.parametrize('atoms', test_materials)
def test_hse(separate_folder, atoms, mockgpaw, mocker):
    import numpy as np
    from pathlib import Path
    atoms.write('structure.json')

    def non_self_consistent_eigenvalues(calc,
                                        xcname,
                                        n1,
                                        n2,
                                        kpt_indices=None,
                                        snapshot=None,
                                        ftol=42.0):
        Path(snapshot).write_text('{}')
        e_skn = calc.eigenvalues[np.newaxis, :, n1:n2]
        return e_skn, e_skn, e_skn

    mocker.patch('gpaw.hybrids.eigenvalues.non_self_consistent_eigenvalues',
                 create=True, new=non_self_consistent_eigenvalues)

    def vxc(calc, xc):
        return calc.eigenvalues[np.newaxis]

    mocker.patch('gpaw.xc.tools.vxc', create=True, new=vxc)
    from asr.hse import main
    main()
    get_webcontent('database.db')
