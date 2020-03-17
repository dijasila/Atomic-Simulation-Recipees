import pytest

from .conftest import test_materials, get_webcontent


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_hse(separate_folder, atoms, mockgpaw, mocker):
    import numpy as np
    from gpaw import GPAW
    from pathlib import Path
    atoms.write("structure.json")

    class EXX:
        def __init__(self, name, xc, bands):
            self.bands = bands
            self.calc = GPAW(name)

        def get_eigenvalue_contributions(self):
            return self.calc.eigenvalues[np.newaxis, :,
                                         self.bands[0]:self.bands[1]]

        def calculate(self, restart=None):
            Path(restart).write_text("{}")

    mocker.patch("gpaw.xc.exx.EXX", create=True, new=EXX)

    def vxc(calc, xc):
        return calc.eigenvalues[np.newaxis]

    mocker.patch("gpaw.xc.tools.vxc", create=True, new=vxc)
    from asr.hse import main
    main()
    get_webcontent('database.db')
