"""Module for calculating formal polarization phase for a structure.

Module for calculating formal polarization phase as defined in the
Modern Theory of Polarization. To learn more see more about this
please see our explanation of the :ref:`Modern theory of
polarization`, in particular to see the definition of the polarization
phase.

The central recipe of this module is :func:`asr.formalpolarization.main`.

.. autofunction:: asr.formalpolarization.main

"""

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


def get_wavefunctions(atoms, name, params):
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from pathlib import Path

    if Path(name).is_file():
        return GPAW(name, communicator=serial_comm, txt=None)

    # We make sure that we converge eigenstates
    convergence = {'eigenstates': 1e-11,
                   'density': 1e-7}
    if 'convergence' in params:
        params['convergence'].update(convergence)
    else:
        params['convergence'] = convergence
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
@option('--gpwname', help='Formal polarization gpw file name.')
@option('--kpts', help='K-point dict for ES calculation.')
def main(gpwname='formalpol.gpw', kpts={'density': 12.0}):
    """Calculate the formal polarization phase.

    Calculate the formal polarization geometric phase necesarry for in
    the modern theory of polarization.
    """
    from pathlib import Path
    from gpaw import GPAW
    from gpaw.mpi import world
    calc = GPAW('gs.gpw', txt=None)
    params = calc.parameters
    params['kpts'] = kpts
    atoms = calc.atoms
    calc = get_wavefunctions(atoms=atoms,
                             name=gpwname,
                             params=params)
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
