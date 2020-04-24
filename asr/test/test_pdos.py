import pytest


# ---------- Mocks ---------- #


class MockProgressbar:
    def __init__(self, *args, **kwargs):
        """Ignore all input arguments."""
        pass

    def enumerate(self, iterable):
        return enumerate(iterable)


def get_number_of_electrons(angular):
    """Make a simple calculation of the number of electrons
    based on the occurence of orbitals."""
    count = 0
    count += 2 * angular.count('s')
    count += 6 * angular.count('p')
    count += 10 * angular.count('d')
    count += 14 * angular.count('f')

    return count


def mock_ldos(calc, a, spin, angular='spdf', *args, **kwargs):
    """Distribute weights on atoms, spins and angular momentum."""
    import numpy as np
    from ase.units import Ha
    from asr.pdos import get_l_a

    # Extract eigenvalues and weights
    e_kn = calc.get_all_eigenvalues() / Ha
    w_k = np.array(calc.get_k_point_weights())

    # Take care of spin degeneracy
    if calc.get_number_of_spins() == 1:
        w_k *= 2

    # Figure out the total number of orbitals to be counted
    l_a = get_l_a(calc.atoms.get_atomic_numbers())
    nstates = 0
    for a, orbitals in l_a.items():
        nstates += get_number_of_electrons(orbitals)
    # Normalize weights
    w_k *= get_number_of_electrons(angular) / nstates

    # Return flattened energies and weights
    nb = e_kn.shape[1]
    return e_kn.flatten(), np.tile(w_k, nb)


# ---------- Tests ---------- #


@pytest.mark.ci
@pytest.mark.parametrize('gap', [0.0, 1.0])
@pytest.mark.parametrize('fermi_level', [0.5, 1.5])
def test_pdos(asr_tmpdir_w_params, mockgpaw, mocker, get_webcontent,
              test_material, gap, fermi_level):
    import numpy as np
    from asr.pdos import main
    from gpaw import GPAW

    # Adjust the gap and the fermi level
    mocker.patch.object(GPAW, '_get_band_gap')
    mocker.patch.object(GPAW, '_get_fermi_level')
    GPAW._get_fermi_level.return_value = fermi_level
    GPAW._get_band_gap.return_value = gap

    # Use mocker for gpaw stuff
    mocker.patch("gpaw.utilities.progressbar.ProgressBar",
                 MockProgressbar)
    mocker.patch("gpaw.utilities.dos.raw_orbital_LDOS",
                 mock_ldos)
    mocker.patch("asr.pdos.raw_spinorbit_orbital_LDOS_hack",
                 mock_ldos)

    # Run recipe
    test_material.write('structure.json')
    results = main()

    # We can check the dos at ef
    dos_at_ef_nosoc = results['dos_at_ef_nosoc']
    # We should be able to compute dos_at_ef manually for the hom. el. gas
    # at different fermi levels, gaps, exchange splittings and dimensionalities
    # This could be a very good ci test!

    # When mocking, spin-orbit coupling should not make a difference
    dos_at_ef_soc = results['dos_at_ef_soc']
    assert dos_at_ef_nosoc == pytest.approx(dos_at_ef_soc)

    # For the pdos, make a sum over spin, symbol and angular momentum
    for pdos_result in [results['pdos_nosoc'], results['pdos_soc']]:
        e_e = pdos_result['energies']
        ef = pdos_result['efermi']
        # Fermi level in pdos calculation should be specified one
        assert ef == pytest.approx(fermi_level)
        # Calculate the sum of pdos as close to the fermi level as possible
        efe = np.argmin(np.abs(e_e - ef))
        dos_at_ef = sum([pdos_e[efe] for _, pdos_e in pdos_result['pdos_syl'].items()])
        # With the mocked raw_orbital_LDOS, total projection weight is 1, so that
        # dos_at_ef should be consistent with the other results (apart from the fact
        # that we are not guaranteed to hit exactly the fermi energy).
        assert dos_at_ef == pytest.approx(dos_at_ef_nosoc, abs=0.1)  # not exactly ef

    # Check content of webpanel
    content = get_webcontent()
    assert ('<td>DensityofstatesattheFermienergyw/osoc.</td>'
            f'<td>{dos_at_ef_nosoc:0.2f}states/(eV*unitcell)</td>') in content
    assert ('<td>DensityofstatesattheFermienergy</td>'
            f'<td>{dos_at_ef_soc:0.2f}states/(eV*unitcell)</td>') in content


# ---------- Integration tests ---------- #


@pytest.mark.integration_test_gpaw
def test_pdos_full(asr_tmpdir_w_params):
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
