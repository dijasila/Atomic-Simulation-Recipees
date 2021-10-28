"""Formal polarization phase.

Module for calculating formal polarization phase as defined in the
Modern Theory of Polarization. To learn more see more about this
please see our explanation of the :ref:`Modern theory of
polarization`, in particular to see the definition of the polarization
phase.

The central recipe of this module is :func:`asr.c2db.formalpolarization.main`.

"""
import numpy as np

from ase import Atoms

import asr

from asr.core import (
    command, ASRResult, atomsopt, calcopt,
    Selector)


class AtomsTooCloseToBoundary(Exception):
    pass


def get_electronic_polarization_phase(calc):
    import numpy as np
    from gpaw.berryphase import get_berry_phases
    from gpaw.mpi import SerialCommunicator
    from ase.parallel import world

    assert isinstance(calc.world, SerialCommunicator)

    phase_c = np.zeros((3,), float)
    # Calculate and save berry phases
    nspins = calc.get_number_of_spins()
    if world.rank == 0:
        for c in [0, 1, 2]:
            for spin in range(nspins):
                _, phases = get_berry_phases(calc, dir=c, spin=spin)
                phase_c[c] += np.sum(phases) / len(phases)

    phase_c = phase_c * 2 / nspins
    world.sum(phase_c)

    return phase_c


def get_atomic_polarization_phase(calc):
    Z_a = []
    for num in calc.atoms.get_atomic_numbers():
        for ida, setup in zip(calc.setups.id_a,
                              calc.setups):
            if abs(ida[0] - num) < 1e-5:
                break
        Z_a.append(setup.Nv)
    phase_c = 2 * np.pi * np.dot(Z_a, calc.spos_ac)
    return phase_c


def get_dipole_polarization_phase(dipole_v, cell_cv):
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    dipole_phase_c = np.dot(B_cv, dipole_v)
    return dipole_phase_c


def get_wavefunctions(atoms, name, calculator):
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from ase.calculators.calculator import get_calculator_class
    calcname = calculator.pop("name")
    calc = get_calculator_class(calcname)(**calculator)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(name, 'all')

    calc = GPAW(name, communicator=serial_comm, txt=None)
    return calc


def distance_to_non_pbc_boundary(atoms, eps=1):
    pbc_c = atoms.get_pbc()
    if pbc_c.all():
        return None
    cell_cv = atoms.get_cell()
    pos_ac = atoms.get_scaled_positions()
    pos_ac -= np.round(pos_ac)
    posnonpbc_av = np.dot(pos_ac[:, ~pbc_c], cell_cv[~pbc_c])
    dist_to_cell_edge_a = np.sqrt((posnonpbc_av**2).sum(axis=1))
    return dist_to_cell_edge_a


sel = Selector()
sel.name = sel.EQ('asr.c2db.formalpolarization:main')
sel.version = sel.EQ(-1)
sel.parameters = sel.CONTAINS('gpwname')


@asr.migration(selector=sel, uid='f4525d8398b44441821e496195081b86')
def remove_gpwname_from_parameters(record):
    """Remove 'gpwname' from parameters."""
    del record.parameters.gpwname
    return record


@command(
    'asr.c2db.formalpolarization',
)
@atomsopt
@calcopt
def main(
        atoms: Atoms,
        calculator: dict = {
            'name': 'gpaw',
            'mode': {'name': 'pw', 'ecut': 800},
            'xc': 'PBE',
            'kpts': {'density': 12.0},
            'occupations': {'name': 'fermi-dirac',
                            'width': 0.05},
            'symmetry': 'off',
            'convergence': {'eigenstates': 1e-11,
                            'density': 1e-7},
            'txt': 'formalpol.txt',
            'charge': 0,
        }) -> ASRResult:
    """Calculate the formal polarization phase.

    Calculate the formal polarization geometric phase necesarry for in
    the modern theory of polarization.
    """
    from pathlib import Path
    from gpaw.mpi import world
    from ase.units import Bohr

    gpwname = 'formalpol.gpw'
    dist_a = distance_to_non_pbc_boundary(atoms)
    if dist_a is not None and np.any(dist_a < 1):
        raise AtomsTooCloseToBoundary(
            'The atoms are too close to a non-pbc boundary '
            'which creates problems when using a dipole correction. '
            f'Please center the atoms in the unit-cell. Distances (Å): {dist_a}.'
        )

    calc = get_wavefunctions(
        atoms=atoms,
        name=gpwname,
        calculator=calculator,
    )
    electronic_phase_c = get_electronic_polarization_phase(calc)
    atomic_phase_c = get_atomic_polarization_phase(calc)
    dipole_v = calc.get_dipole_moment() / Bohr
    cell_cv = atoms.get_cell() / Bohr
    dipole_phase_c = get_dipole_polarization_phase(dipole_v, cell_cv)

    # Total phase
    pbc_c = atoms.get_pbc()
    phase_c = electronic_phase_c + atomic_phase_c
    phase_c[~pbc_c] = dipole_phase_c[~pbc_c]

    results = {'phase_c': phase_c,
               'electronic_phase_c': electronic_phase_c,
               'atomic_phase_c': atomic_phase_c,
               'dipole_phase_c': dipole_phase_c,
               'dipole_v': dipole_v}
    world.barrier()
    if world.rank == 0:
        f = Path(gpwname)
        if f.is_file():
            f.unlink()

    return results


if __name__ == '__main__':
    main.cli()
