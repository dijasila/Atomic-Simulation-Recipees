import pytest

from .conftest import test_materials, freeelectroneigenvalues, get_webcontent


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_gw(separate_folder, atoms, mockcalculator, mocker):
    import numpy as np
    import gpaw
    from gpaw.response.g0w0 import G0W0

    get_eigenvalues = freeelectroneigenvalues(atoms, gap=1)
    mocker.patch.object(gpaw.GPAW, "get_eigenvalues", new=get_eigenvalues)
    mocker.patch.object(gpaw.GPAW, "get_fermi_level")
    gpaw.GPAW.get_fermi_level.return_value = 0.5

    from asr.gw import main
    atoms.write("structure.json")
    ndim = sum(atoms.pbc)

    def calculate(self):
        self.calc.get_eigenvalues(kpt=0)
        eps = self.calc.tmpeigenvalues[np.newaxis, :, 0:6]

        return {"qp": eps,
                "Z": np.zeros_like(eps) + 1,
                "eps": eps}

    mocker.patch.object(G0W0, "calculate", calculate)
    if ndim > 1:
        main()
        get_webcontent('database.db')
    else:
        with pytest.raises(NotImplementedError):
            main()
