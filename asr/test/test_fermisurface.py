import pytest
from ase.units import Hartree, Bohr
import numpy as np
from .materials import BN


@pytest.mark.ci
def test_fermisurface(
        asr_tmpdir, mockgpaw, mocker, get_webcontent,
):
    from asr.fermisurface import main
    import gpaw
    fermi_level = 0.5
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = fermi_level
    BN.write('structure.json')
    result = main()

    fermi_wave_vector = (2 * fermi_level / Hartree)**0.5 / Bohr

    moduli = np.sqrt(np.sum(result.contours[:, :2]**2, axis=1))
    assert moduli == pytest.approx(fermi_wave_vector, rel=0.01)

    content = get_webcontent()
    assert 'fermi' in content
