import pytest
from asr.c2db.pdos import PDOSWorkflow
from asr.c2db.gs import GSWorkflow


def workflow(rn, atoms, calculator, **kw):
    gsw = GSWorkflow(rn, atoms=atoms, calculator=calculator)
    pdosw = PDOSWorkflow(rn, gsworkflow=gsw, **kw)
    return pdosw.postprocess


@pytest.mark.ci
def test_pdos(repo, mockgpaw,
              test_material, get_webcontent, fast_calc):

    pdospost = repo.run_workflow_blocking(workflow, atoms=test_material,
                                          calculator=fast_calc,
                                          emptybands=5,
                                          kptdensity=2)

    # PDOS(atoms=test_material, calculator=fast_calc, emptybands=5, kptdensity=2.0)

    # main(atoms=test_material, calculator=fast_calc)
    # test_material.write('structure.json')
    # get_webcontent()


@pytest.mark.integration_test_gpaw
def test_pdos_full(asr_tmpdir_w_params):
    from pathlib import Path
    import numpy as np
    from ase.build import bulk
    from ase.dft.kpoints import monkhorst_pack
    from gpaw import GPAW, PW
    from asr.core import write_json
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

        Li1.calc = calc1
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

        Li2.calc = calc2
        Li2.get_potential_energy()

        calc2.write('Li2.gpw')

    # Part 2: density of states at the fermi level

    # Dump json file to fake magnetic_anisotropy recipe
    dct = {'theta': theta, 'phi': phi}
    write_json('results-asr.c2db.magnetic_anisotropy.json', dct)

    def dosef(dos, spin=None):
        return dos.raw_dos([0.0], spin, 0.0)[0]

    # Calculate the dos at ef for each spin channel
    # spin-0
    dos1 = calc1.dos()
    dosef10 = dosef(dos1) / 2
    # spin-polarized
    from gpaw.dos import DOSCalculator
    dos2 = DOSCalculator.from_calculator(calc2)
    dosef20 = dosef(dos2, spin=0)
    dosef21 = dosef(dos2, spin=1)

    # Calculate the dos at ef with soc using asr
    # spin-0
    dosef_soc1 = dosef(calc1.dos(soc=True))
    # spin-polarized
    dosef_soc2 = dosef(calc2.dos(soc=True))

    # Part 3: test output values

    # Test ase
    dosef_d = np.array([dosef10, dosef20, dosef21])
    assert np.all(np.abs(dosef_d - dos0) < dos0tol),\
        ("ASE doesn't reproduce single spin dosef: "
         f"{dosef_d}, {dos0}")

    assert abs(dosef10 * 2 - dosef_soc1) < dos_socnosoc_eqtol,\
        ("ASR's nosoc/soc methodology disagrees in the spin-0 case: "
         f"{dosef10}, {dosef_soc1}")
    assert abs(dosef_soc1 - dosef_soc2) < dos0tol,\
        ("ASR's nosoc/soc methodology disagrees in the spin-polarized case: "
         f"{dosef_soc1}, {dosef_soc2}")

    print('All good')
