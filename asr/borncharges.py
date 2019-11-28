from asr.core import command, option


def get_polarization_phase(calc):
    import numpy as np
    from gpaw.berryphase import get_berry_phases
    from gpaw.mpi import SerialCommunicator

    assert isinstance(calc.world, SerialCommunicator)

    phase_c = np.zeros((3,), float)
    # Calculate and save berry phases
    nspins = calc.wfs.nspins
    for c in [0, 1, 2]:
        for spin in range(nspins):
            indices_kk, phases = get_berry_phases(calc, dir=c, spin=spin)
            phase_c[c] += np.sum(phases) / len(phases)

    # Ionic contribution
    Z_a = []
    for num in calc.atoms.get_atomic_numbers():
        for ida, setup in zip(calc.wfs.setups.id_a,
                              calc.wfs.setups):
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
    tmp = Path(name).with_suffix('').name
    calc = GPAW(txt=tmp + '.txt', **params)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write(name, 'all')

    calc = GPAW(name, communicator=serial_comm, txt=None)
    return calc


def webpanel(row, key_descriptions):
    import numpy as np

    def matrixtable(M, digits=2, unit='', skiprow=0, skipcolumn=0):
        table = M.tolist()
        shape = M.shape

        for i in range(skiprow, shape[0]):
            for j in range(skipcolumn, shape[1]):
                value = table[i][j]
                table[i][j] = '{:.{}f}{}'.format(value, digits, unit)
        return table

    columns = [[], []]
    for a, Z_vv in enumerate(
            row.data['results-asr.borncharges.json']['Z_avv']):
        table = np.zeros((4, 4))
        table[1:, 1:] = Z_vv
        rows = matrixtable(table, skiprow=1, skipcolumn=1)
        sym = row.symbols[a]
        rows[0] = [f'Z<sup>{sym}</sup><sub>ij</sub>', 'u<sub>x</sub>',
                   'u<sub>y</sub>', 'u<sub>z</sub>']
        rows[1][0] = 'P<sub>x</sub>'
        rows[2][0] = 'P<sub>y</sub>'
        rows[3][0] = 'P<sub>y</sub>'

        Ztable = dict(
            type='table',
            rows=rows)

        columns[a % 2].append(Ztable)

    panel = {'title': 'Born charges',
             'columns': columns}
    return [panel]


@command('asr.borncharges',
         dependencies=['asr.gs@calculate'],
         requires=['gs.gpw'],
         webpanel=webpanel)
@option('--displacement', help='Atomic displacement (Å)')
@option('--kptdensity')
def main(displacement=0.01, kptdensity=8.0):
    """Calculate Born charges"""
    from pathlib import Path

    import numpy as np
    from gpaw import GPAW
    from gpaw.mpi import world

    from ase.parallel import parprint
    from ase.units import Bohr

    from asr.core import file_barrier

    calc = GPAW('gs.gpw', txt=None)
    params = calc.parameters
    atoms = calc.atoms
    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    pos_av = atoms.get_positions().copy()
    Z_avv = []
    P_asvv = []

    parprint('Atomnum Atom Direction Displacement')
    for a in range(len(atoms)):
        phase_scv = np.zeros((2, 3, 3), float)
        for v in range(3):
            for s, sign in enumerate([-1, 1]):
                if world.rank == 0:
                    print(sym_a[a], a, v, s)
                # Update atomic positions
                atoms.positions = pos_av
                atoms.positions[a, v] = pos_av[a, v] + sign * displacement
                prefix = 'born-{}-{}{}{}'.format(displacement, a,
                                                 'xyz'[v],
                                                 ' +-'[sign])
                name = prefix + '.gpw'
                calc = get_wavefunctions(atoms, name, params,
                                         density=kptdensity)
                try:
                    phase_c = get_polarization_phase(calc)
                except ValueError:
                    with file_barrier(name):
                        calc = get_wavefunctions(atoms, name,
                                                 params,
                                                 density=kptdensity)
                        phase_c = get_polarization_phase(name)

                phase_scv[s, :, v] = phase_c

        dphase_cv = (phase_scv[1] - phase_scv[0])
        mod_cv = np.round(dphase_cv / (2 * np.pi)) * 2 * np.pi
        dphase_cv -= mod_cv
        phase_scv[1] -= mod_cv
        dP_vv = (-np.dot(dphase_cv.T, cell_cv).T /
                 (2 * np.pi * vol))

        P_svv = (-np.dot(cell_cv.T, phase_scv).transpose(1, 0, 2) /
                 (2 * np.pi * vol))
        Z_vv = dP_vv * vol / (2 * displacement / Bohr)
        P_asvv.append(P_svv)
        Z_avv.append(Z_vv)

    data = {'Z_avv': Z_avv, 'sym_a': sym_a,
            'P_asvv': P_asvv}

    # world.barrier()
    # if world.rank == 0:
    #     files = Path().glob('born-*.gpw')
    #     for f in files:
    #         f.unlink()

    return data


if __name__ == '__main__':
    main.cli()
