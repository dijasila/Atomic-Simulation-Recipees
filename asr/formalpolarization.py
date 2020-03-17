from asr.core import command, option


def get_polarization_phase(calc):
    import numpy as np
    from gpaw.berryphase import get_berry_phases
    from gpaw.mpi import SerialCommunicator

    assert isinstance(calc.world, SerialCommunicator)

    phase_c = np.zeros((3,), float)
    # Calculate and save berry phases
    nspins = calc.get_number_of_spins()
    for c in [0, 1, 2]:
        for spin in range(nspins):
            _, phases = get_berry_phases(calc, dir=c, spin=spin)
            phase_c[c] += np.sum(phases) / len(phases)

    # Ionic contribution
    Z_a = []
    for num in calc.atoms.get_atomic_numbers():
        for ida, setup in zip(calc.setups.id_a,
                              calc.setups):
            if abs(ida[0] - num) < 1e-5:
                break
        Z_a.append(setup.Nv)

    phase_c = phase_c * 2 / nspins
    phase_c += 2 * np.pi * np.dot(Z_a, calc.spos_ac)

    return -phase_c


def get_wavefunctions(atoms, name, params, density=6.0,
                      no_symmetries=False):
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from pathlib import Path

    if Path(name).is_file():
        return GPAW(name, communicator=serial_comm, txt=None)

    params['kpts'] = {'density': density,
                      'gamma': True}
    # 'even': True}  # Not compatible with ASE atm.
    if no_symmetries:
        params['symmetry'] = {'point_group': False,
                              'time_reversal': False}
    else:
        params['symmetry'] = {'point_group': True,
                              'time_reversal': True}
    params['convergence']['eigenstates'] = 1e-11
    params['convergence']['density'] = 1e-7
    tmp = Path(name).with_suffix('').name
    params['txt'] = tmp + '.txt'
    calc = GPAW(**params)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write(name, 'all')

    calc = GPAW(name, communicator=serial_comm, txt=None)
    return calc


@command(dependencies=['asr.gs@calculate'],
         requires=['gs.gpw'])
@option('--gpwname', help='Formal polarization gpw file name')
@option('--kptdensity', help='Kpoint density for gpw file')
def main(gpwname='formalpol.gpw', kptdensity=12.0):
    from pathlib import Path
    from gpaw import GPAW
    from gpaw.mpi import world
    calc = GPAW('gs.gpw', txt=None)
    params = calc.parameters
    atoms = calc.atoms
    calc = get_wavefunctions(atoms=atoms, name=gpwname,
                             params=params, density=kptdensity)
    phase_c = get_polarization_phase(calc)
    results = {'phase_c': phase_c}
    world.barrier()
    if world.rank == 0:
        f = Path(gpwname)
        if f.is_file():
            f.unlink()

    return results


if __name__ == '__main__':
    main.cli()
