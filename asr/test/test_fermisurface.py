import pytest
from ase.units import Hartree, Bohr
import numpy as np
from .materials import BN


@pytest.mark.ci
def test_fermisurface(
        in_tempdir,
        mockgpaw,
        mocker,
        get_webcontent,
        fast_calc,
):
    from asr.c2db.fermisurface import main
    from asr.c2db.gs import GS
    import gpaw
    fermi_level = 0.5
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = fermi_level

    gs = GS(atoms=BN, calculator=fast_calc)
    result = main(gsresult=gs.gsresult, mag_ani=gs.mag_ani)

    fermi_wave_vector = (2 * fermi_level / Hartree)**0.5 / Bohr

    moduli = np.sqrt(np.sum(result.contours[:, :2]**2, axis=1))
    assert moduli == pytest.approx(fermi_wave_vector, rel=0.1)
    # BN.write('structure.json')
    # content = get_webcontent()
    # assert 'fermi' in content
