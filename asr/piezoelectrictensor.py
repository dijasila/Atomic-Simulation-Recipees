from asr.core import command, option


def webpanel(row, key_descriptions):
    for i in range(1, 4):
        for j in range(1, 7):
            key = 'e0_{}{}'.format(i, j)
            name = 'Clamped piezoelectric tensor'
            description = ('{} ({}{})'.format(name, i, j),
                           '{} ({}{}-component)'.format(name, i, j),
                           '`\\text{Ang}^{-1}`')
            key_descriptions[key] = description

            key = 'e_{}{}'.format(i, j)
            name = 'Piezoelectric tensor'
            description = ('{} ({}{})'.format(name, i, j),
                           '{} ({}{}-component)'.format(name, i, j),
                           '`\\text{Ang}^{-1}`')
            key_descriptions[key] = description

    if 'e_vvv' in row.data:
        def matrixtable(M, digits=2):
            table = M.tolist()
            shape = M.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = table[i][j]
                    table[i][j] = '{:.{}f}'.format(value, digits)
            return table

        e_ij = row.data.e_vvv[:, [0, 1, 2, 1, 0, 0],
                              [0, 1, 2, 2, 2, 1]]
        e0_ij = row.data.e0_vvv[:, [0, 1, 2, 1, 0, 0],
                                [0, 1, 2, 2, 2, 1]]

        etable = dict(
            header=['Piezoelectric tensor', '', ''],
            type='table',
            rows=matrixtable(e_ij))

        e0table = dict(
            header=['Clamped piezoelectric tensor', ''],
            type='table',
            rows=matrixtable(e0_ij))

        columns = [[etable, e0table], []]

        panel = [('Piezoelectric tensor', columns)]
    else:
        panel = ()
    things = ()
    return panel, things


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
    params['convergence']['density'] = 1e-7
    tmp = Path(name).with_suffix('').name
    calc = GPAW(txt=tmp + '.txt', **params)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write(name, 'all')

    calc = GPAW(name, communicator=serial_comm, txt=None)
    return calc

@command()
def main(delta=0.01):
    import numpy as np
    from gpaw import GPAW
    calc = GPAW('gs.gpw', txt=None)
    params = calc.parameters

    # Do not symmetrize the density
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True,
                          'time_reversal': False}
    # We need the eigenstates to a higher accuracy
    params['convergence']['density'] = 1e-8
    atoms = calc.atoms

    # From experience it is important to use
    # non-gamma centered grid when using symmetries.
    # Might have something to do with degeneracies, not sure.
    size = kptdensity2monkhorstpack(atoms, 6.0, True)
    params['kpts'] = {'size': size, 'gamma': False}
    oldcell_cv = atoms.get_cell()
    pbc_c = atoms.pbc

    if world.rank == 0:
        print('i j s')
    epsclamped_vvv = np.zeros((3, 3, 3), float)
    eps_vvv = np.zeros((3, 3, 3), float)

    for i in range(3):
        for j in range(3):
            if j < i:
                continue
            phaseclamped_sc = np.zeros((2, 3), float)
            phase_sc = np.zeros((2, 3), float)
            if world.rank == 0:
                print(i, j)
            for s, sign in enumerate([-1, 1]):
                if not pbc_c[i] or not pbc_c[j]:
                    continue

                # Update atomic structure
                strain_vv = np.zeros((3, 3), float)
                strain_vv[i, j] = sign * delta
                newcell_cv = np.dot(oldcell_cv,
                                    np.eye(3) + strain_vv)
                atoms.set_cell(newcell_cv, scale_atoms=True)
                namegpw = 'piezo-{}-{}{}{}.gpw'.format(delta, i, j, '-+'[s])
                berryname = 'piezo-{}-{}{}{}-berryphases.json'.format(delta, i,
                                                                      j,
                                                                      '-+'[s])
                if not exists(namegpw) and not exists(berryname):
                    calc = get_wavefunctions(atoms, namegpw, params)

                try:
                    phaseclamped_sc[s] = get_polarization_phase(namegpw)
                except ValueError:
                    calc = get_wavefunctions(atoms, namegpw, params)
                    phaseclamped_sc[s] = get_polarization_phase(namegpw)

                # Now relax atoms
                relaxname = 'piezo-{}-{}{}{}-relaxed'.format(delta, i,
                                                             j, '-+'[s])
                relaxnamegpw = relaxname + '.gpw'
                berryname = relaxname + '-berryphases.json'
                if not exists(relaxnamegpw) and not exists(berryname):
                    try:
                        relaxedatoms = read(relaxname + '.traj')
                    except (IOError, UnknownFileTypeError, JSONDecodeError):
                        relaxedatoms = atoms.copy()
                    constraint = FixAtoms(indices=[0])
                    relaxedatoms.set_constraint(constraint)
                    relaxedatoms.calc = GPAW(txt=relaxname + '.txt', **params)
                    opt = BFGS(relaxedatoms,
                               logfile=relaxname + '.log',
                               trajectory=relaxname + '.traj')
                    opt.run(fmax=0.001, smax=0.0002,
                            smask=[0, 0, 0, 0, 0, 0],
                            emin=-np.inf)
                    relaxedatoms = read(relaxname + '.traj')
                    calc = get_wavefunctions(relaxedatoms, relaxnamegpw,
                                             params)

                try:
                    phase_sc[s] = get_polarization_phase(relaxnamegpw)
                except ValueError:
                    calc = get_wavefunctions(relaxedatoms, relaxnamegpw,
                                             params)
                    phase_sc[s] = get_polarization_phase(relaxnamegpw)

            world.barrier()
            if world.rank == 0:
                files = glob('piezo-{}-*.gpw'.format(delta))
                for f in files:
                    if isfile(f):
                        remove(f)

            vol = abs(np.linalg.det(oldcell_cv / Bohr))
            dphase_c = phaseclamped_sc[1] - phaseclamped_sc[0]
            dphase_c -= np.round(dphase_c / (2 * np.pi)) * 2 * np.pi
            dphasedeps_c = dphase_c / (2 * delta)
            eps_v = (-np.dot(dphasedeps_c, oldcell_cv / Bohr) /
                     (2 * np.pi * vol))

            if (~atoms.pbc).any():
                L = np.abs(np.linalg.det(oldcell_cv[~pbc_c][:, ~pbc_c] / Bohr))
                eps_v *= L

            epsclamped_vvv[:, i, j] = eps_v
            epsclamped_vvv[:, j, i] = eps_v
            if world.rank == 0:
                print('phaseclamped_sc', np.round(phaseclamped_sc, 5))
                print('dphase_c', np.round(dphase_c, 5))
                print('Clamped eps_v', np.round(eps_v, 5))

            dphase_c = phase_sc[1] - phase_sc[0]
            dphase_c -= np.round(dphase_c / (2 * np.pi)) * 2 * np.pi
            dphasedeps_c = dphase_c / (2 * delta)
            eps_v = (-np.dot(dphasedeps_c, oldcell_cv / Bohr) /
                     (2 * np.pi * vol))
            if (~atoms.pbc).any():
                eps_v *= L

            eps_vvv[:, i, j] = eps_v
            eps_vvv[:, j, i] = eps_v
            if world.rank == 0:
                print('phase_sc', np.round(phase_sc, 5))
                print('dphase_c', np.round(dphase_c, 5))
                print('Clamped eps_v', np.round(eps_v, 5))

    if world.rank == 0:
        print('epsclamped_vv')
        print(np.round(epsclamped_vvv[:,
                                      [0, 1, 2, 1, 0, 0],
                                      [0, 1, 2, 2, 2, 1]], 3))
        print('epsclamped_vv nC / m')
        print(np.round(epsclamped_vvv[:,
                                      [0, 1, 2, 1, 0, 0],
                                      [0, 1, 2, 2, 2, 1]] *
                       1.602e-19 / (Bohr * 1e-10) * 1e9, 3))

        print('eps_vv')
        print(np.round(eps_vvv[:,
                               [0, 1, 2, 1, 0, 0],
                               [0, 1, 2, 2, 2, 1]], 3))
        print('eps_vv nC / m')
        print(np.round(eps_vvv[:,
                               [0, 1, 2, 1, 0, 0],
                               [0, 1, 2, 2, 2, 1]] *
                       1.602e-19 / (Bohr * 1e-10) * 1e9, 3))

    data = {'eps_vvv': eps_vvv,
            'epsclamped_vvv': epsclamped_vvv}

    filename = 'piezoelectrictensor-{}.json'.format(delta)

    with paropen(filename, 'w') as fd:
        json.dump(jsonio.encode(data), fd)

    world.barrier()
    if world.rank == 0:
        files = glob('piezo-{}-*.gpw'.format(delta))
        for f in files:
            if isfile(f):
                remove(f)


if __name__ == '__main__':
    main.cli()
