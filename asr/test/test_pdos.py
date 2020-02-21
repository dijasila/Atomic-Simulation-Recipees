import pytest
from .conftest import test_materials, get_webcontent


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_pdos(separate_folder, mockgpaw, mocker, atoms):
    from asr.pdos import main

    mocker.patch("gpaw.utilities.dos.raw_orbital_LDOS", create=True)
    mocker.patch("gpaw.utilities.progressbar.ProgressBar", create=True)
    atoms.write('structure.json')
    main()
    get_webcontent()


@pytest.mark.integration_test_gpaw
def test_pdos_full(separate_folder):
    from pathlib import Path
    import numpy as np

    from ase.build import bulk
    from ase.dft.kpoints import monkhorst_pack
    from ase.dft.dos import DOS

    from asr.pdos import dos_at_ef
    from asr.core import write_json
    from gpaw import GPAW, PW
    # ------------------- Inputs ------------------- #

    # Part 1: ground state calculation
    xc = 'LDA'
    kpts = 9
    nb = 5
    pw = 300
    a = 3.51
    mm = 0.001

    # Part 2: density of states at the fermi energy
    theta = 0.
    phi = 0.

    # Part 3: test output values
    dos0 = 0.274
    dos0tol = 0.01
    dos_eqtol = 0.01
    dos_socnosoc_eqtol = 0.1

    # ------------------- Script ------------------- #

    # Part 1: ground state calculation

    # spin-0 calculation
    if Path('Li1.gpw').is_file():
        calc1 = GPAW('Li1.gpw', txt=None)
    else:
        Li1 = bulk('Li', 'bcc', a=a)
        calc1 = GPAW(xc=xc,
                     mode=PW(pw),
                     kpts=monkhorst_pack((kpts, kpts, kpts)),
                     nbands=nb,
                     idiotproof=False)

        Li1.set_calculator(calc1)
        Li1.get_potential_energy()

        calc1.write('Li1.gpw')

    # spin-polarized calculation
    if Path('Li2.gpw').is_file():
        calc2 = GPAW('Li2.gpw', txt=None)
    else:
        Li2 = bulk('Li', 'bcc', a=a)
        Li2.set_initial_magnetic_moments([mm])

        calc2 = GPAW(xc=xc,
                     mode=PW(pw),
                     kpts=monkhorst_pack((kpts, kpts, kpts)),
                     nbands=nb,
                     idiotproof=False)

        Li2.set_calculator(calc2)
        Li2.get_potential_energy()

        calc2.write('Li2.gpw')

    # Part 2: density of states at the fermi level

    # Dump json file to fake magnetic_anisotropy recipe
    dct = {'theta': theta, 'phi': phi}
    write_json('results-asr.magnetic_anisotropy.json', dct)

    # Calculate the dos at ef for each spin channel
    # spin-0
    dos1 = DOS(calc1, width=0., window=(-0.1, 0.1), npts=3)
    dosef10 = dos1.get_dos(spin=0)[1]
    dosef11 = dos1.get_dos(spin=1)[1]
    # spin-polarized
    dos2 = DOS(calc2, width=0., window=(-0.1, 0.1), npts=3)
    dosef20 = dos2.get_dos(spin=0)[1]
    dosef21 = dos2.get_dos(spin=1)[1]

    # Calculate the dos at ef w/wo soc using asr
    # spin-0
    dosef_nosoc1 = dos_at_ef(calc1, 'Li1.gpw', soc=False)
    dosef_soc1 = dos_at_ef(calc1, 'Li1.gpw', soc=True)
    # spin-polarized
    dosef_nosoc2 = dos_at_ef(calc2, 'Li2.gpw', soc=False)
    dosef_soc2 = dos_at_ef(calc2, 'Li2.gpw', soc=True)

    # Part 3: test output values

    # Test ase
    dosef_d = np.array([dosef10, dosef11, dosef20, dosef21])
    assert np.all(np.abs(dosef_d - dos0) < dos0tol),\
        ("ASE doesn't reproduce single spin dosef: "
         f"{dosef_d}, {dos0}")

    # Test asr
    assert abs(dosef10 + dosef11 - dosef_nosoc1) < dos_eqtol,\
        ("ASR doesn't reproduce ASE's dosef_nosoc in the spin-0 case: "
         f"{dosef10}, {dosef11}, {dosef_nosoc1}")
    assert abs(dosef20 + dosef21 - dosef_nosoc2) < dos_eqtol,\
        ("ASR doesn't reproduce ASE's dosef_nosoc in the spin-polarized case: "
         f"{dosef20}, {dosef21}, {dosef_nosoc2}")
    assert abs(dosef_nosoc1 - dosef_soc1) < dos_socnosoc_eqtol,\
        ("ASR's nosoc/soc methodology disagrees in the spin-0 case: "
         f"{dosef_nosoc1}, {dosef_soc1}")
    assert abs(dosef_nosoc2 - dosef_soc2) < dos_socnosoc_eqtol,\
        ("ASR's nosoc/soc methodology disagrees in the spin-polarized case: "
         f"{dosef_nosoc2}, {dosef_soc2}")

    print('All good')
