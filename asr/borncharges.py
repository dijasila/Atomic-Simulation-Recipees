from asr.utils import command, option


def get_overlap(calc, bands, u1_nR, u2_nR, P1_ani, P2_ani, dO_aii, bG_v):
    import numpy as np
    M_nn = np.dot(u1_nR.conj(), u2_nR.T) * calc.wfs.gd.dv
    cell_cv = calc.wfs.gd.cell_cv
    r_av = np.dot(calc.spos_ac, cell_cv)

    for ia in range(len(P1_ani)):
        P1_ni = P1_ani[ia][bands]
        P2_ni = P2_ani[ia][bands]
        phase = np.exp(-1.0j * np.dot(bG_v, r_av[ia]))
        dO_ii = dO_aii[ia]
        M_nn += P1_ni.conj().dot(dO_ii).dot(P2_ni.T) * phase

    return M_nn


def get_berry_phases(calc, spin=0, dir=0, check2d=False):
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
    import numpy as np

    if isinstance(calc, str):
        calc = GPAW(calc, communicator=serial_comm, txt=None)

    M = np.round(calc.get_magnetic_moment())
    assert np.allclose(M, calc.get_magnetic_moment(), atol=0.05), \
        print(M, calc.get_magnetic_moment())
    nvalence = calc.wfs.setups.nvalence
    nocc_s = [int((nvalence + M) / 2), int((nvalence - M) / 2)]
    assert np.allclose(np.sum(nocc_s), nvalence)
    nocc = nocc_s[spin]

    bands = list(range(nocc))
    kpts_kc = calc.get_bz_k_points()
    size = get_monkhorst_pack_size_and_offset(kpts_kc)[0]
    Nk = len(kpts_kc)
    wfs = calc.wfs
    icell_cv = (2 * np.pi) * np.linalg.inv(calc.wfs.gd.cell_cv).T

    dO_aii = []
    for ia, id in enumerate(wfs.setups.id_a):
        dO_ii = calc.wfs.setups[ia].dO_ii
        dO_aii.append(dO_ii)

    kd = calc.wfs.kd
    nik = kd.nibzkpts

    u_knR = []
    P_kani = []
    for k in range(Nk):
        ik = kd.bz2ibz_k[k]
        k_c = kd.bzk_kc[k]
        ik_c = kd.ibzk_kc[ik]
        kpt = wfs.kpt_u[ik + spin * nik]
        psit_nG = kpt.psit_nG
        ut_nR = wfs.gd.empty(nocc, wfs.dtype)

        # Check that all states are occupied
        assert np.all(kpt.f_n[:nocc] > 1e-6)
        sym = kd.sym_k[k]
        U_cc = kd.symmetry.op_scc[sym]
        time_reversal = kd.time_reversal_k[k]
        sign = 1 - 2 * time_reversal
        phase_c = k_c - sign * np.dot(U_cc, ik_c)
        phase_c = phase_c.round().astype(int)

        N_c = wfs.gd.N_c

        if (U_cc == np.eye(3)).all() or np.allclose(ik_c - k_c, 0.0):
            for n in range(nocc):
                ut_nR[n, :] = wfs.pd.ifft(psit_nG[n], ik)
        else:
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')

            for n in range(nocc):
                ut_nR[n, :] = wfs.pd.ifft(psit_nG[n],
                                          ik).ravel()[i].reshape(N_c)

        if time_reversal:
            ut_nR = ut_nR.conj()

        if np.any(phase_c):
            emikr_R = np.exp(-2j * np.pi *
                             np.dot(np.indices(N_c).T, phase_c / N_c).T)
            u_knR.append(ut_nR * emikr_R[np.newaxis])
        else:
            u_knR.append(ut_nR)

        a_a = []
        U_aii = []
        for a, id in enumerate(wfs.setups.id_a):
            b = kd.symmetry.a_sa[sym, a]
            S_c = np.dot(calc.spos_ac[a], U_cc) - calc.spos_ac[b]
            x = np.exp(2j * np.pi * np.dot(k_c, S_c))
            U_ii = wfs.setups[a].R_sii[sym].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        P_ani = []
        for b, U_ii in zip(a_a, U_aii):
            P_ni = np.dot(kpt.P_ani[b][:nocc], U_ii)
            if time_reversal:
                P_ni = P_ni.conj()
            P_ani.append(P_ni)

        P_kani.append(P_ani)

    indices_kkk = np.arange(Nk).reshape(size)
    tmp = np.concatenate([[i for i in range(3) if i != dir], [dir]])
    indices_kk = indices_kkk.transpose(tmp).reshape(-1, size[dir])

    nkperp = len(indices_kk)
    phases = []
    if check2d:
        phases2d = []
    for indices_k in indices_kk:
        M_knn = []
        for j in range(size[dir]):
            k1 = indices_k[j]
            G_c = np.array([0, 0, 0])
            if j + 1 < size[dir]:
                k2 = indices_k[j + 1]
            else:
                k2 = indices_k[0]
                G_c[dir] = 1
            u1_nR = u_knR[k1]
            u2_nR = u_knR[k2]
            k1_c = kpts_kc[k1]
            k2_c = kpts_kc[k2] + G_c

            if np.any(G_c):
                emiGr_R = np.exp(-2j * np.pi *
                                 np.dot(np.indices(N_c).T, G_c / N_c).T)
                u2_nR = u2_nR * emiGr_R

            bG_c = k2_c - k1_c
            bG_v = np.dot(bG_c, icell_cv)
            M_nn = get_overlap(calc,
                               bands,
                               np.reshape(u1_nR, (nocc, -1)),
                               np.reshape(u2_nR, (nocc, -1)),
                               P_kani[k1],
                               P_kani[k2],
                               dO_aii,
                               bG_v)
            M_knn.append(M_nn)
        det = np.linalg.det(M_knn)
        phases.append(np.imag(np.log(np.prod(det))))
        if check2d:
            # In the case of 2D systems we can check the
            # result
            k1 = indices_k[0]
            k1_c = kpts_kc[k1]
            G_c = [0, 0, 1]
            G_v = np.dot(G_c, icell_cv)
            u1_nR = u_knR[k1]
            emiGr_R = np.exp(-2j * np.pi *
                             np.dot(np.indices(N_c).T, G_c / N_c).T)
            u2_nR = u1_nR * emiGr_R

            M_nn = get_overlap(calc,
                               bands,
                               np.reshape(u1_nR, (nocc, -1)),
                               np.reshape(u2_nR, (nocc, -1)),
                               P_kani[k1],
                               P_kani[k1],
                               dO_aii,
                               G_v)
            phase2d = np.imag(np.log(np.linalg.det(M_nn)))
            phases2d.append(phase2d)

    # Make sure the phases are continuous
    for p in range(nkperp - 1):
        delta = phases[p] - phases[p + 1]
        phases[p + 1] += np.round(delta / (2 * np.pi)) * 2 * np.pi

    phase = np.sum(phases) / nkperp
    if check2d:
        for p in range(nkperp - 1):
            delta = phases2d[p] - phases2d[p + 1]
            phases2d[p + 1] += np.round(delta / (2 * np.pi)) * 2 * np.pi

        phase2d = np.sum(phases2d) / nkperp

        diff = abs(phase - phase2d)
        if diff > 0.01:
            msg = 'Warning wrong phase: phase={}, 2dphase={}'
            print(msg.format(phase, phase2d))

    return indices_kk, phases


def get_polarization_phase(calc):
    from os.path import splitext, exists
    import numpy as np
    from gpaw.mpi import world, serial_comm
    from gpaw import GPAW
    import json

    assert isinstance(calc, str)
    name = splitext(calc)[0]
    berryname = '{}-berryphases.json'.format(name)
    
    phase_c = np.zeros((3,), float)
    if not exists(berryname) and world.rank == 0:
        # Calculate and save berry phases
        calc = GPAW(calc, communicator=serial_comm, txt=None)
        nspins = calc.wfs.nspins
        data = {}
        for c in [0, 1, 2]:
            data[c] = {}
            for spin in range(nspins):
                indices_kk, phases = get_berry_phases(calc, dir=c, spin=spin)
                data[c][spin] = phases

        # Ionic contribution
        Z_a = []
        for num in calc.atoms.get_atomic_numbers():
            for ida, setup in zip(calc.wfs.setups.id_a,
                                  calc.wfs.setups):
                if abs(ida[0] - num) < 1e-5:
                    break
            Z_a.append(setup.Nv)
        data['Z_a'] = Z_a
        data['spos_ac'] = calc.spos_ac.tolist()

        with open(berryname, 'w') as fd:
            json.dump(data, fd, indent=True)

    world.barrier()
    # Read data and calculate phase
    if world.rank == 0:
        print('Reading berryphases {}'.format(berryname))
    with open(berryname) as fd:
        data = json.load(fd)
    
    for c in [0, 1, 2]:
        nspins = len(data[str(c)])
        for spin in range(nspins):
            phases = data[str(c)][str(spin)]
            phase_c[c] += np.sum(phases) / len(phases)
    phase_c = phase_c * 2 / nspins

    Z_a = data['Z_a']
    spos_ac = data['spos_ac']
    phase_c += 2 * np.pi * np.dot(Z_a, spos_ac)
    
    return phase_c


def get_wavefunctions(atoms, name, params, density=6.0,
                      no_symmetries=False):
    from gpaw import GPAW
    from pathlib import Path

    params['kpts'] = {'density': density,
                      'gamma': True,
                      'even': True}
    if no_symmetries:
        params['symmetry'] = {'point_group': False,
                              'time_reversal': False}
    else:
        params['symmetry'] = {'point_group': True,
                              'time_reversal': True}
    params['convergence']['eigenstates'] = 1e-11
    tmp = Path(name).with_suffix('').name
    atoms.calc = GPAW(txt=tmp + '.txt', **params)
    atoms.get_potential_energy()
    atoms.calc.write(name, 'all')
    return atoms.calc


@command('asr.borncharges',
         dependencies=['asr.structureinfo', 'asr.gs'],
         resources='24:10h')
@option('--displacement', help='Atomic displacement (Å)')
@option('--kptdensity')
@option('--folder')
def main(displacement=0.01, kptdensity=6.0, folder='data-borncharges'):
    """Calculate Born charges"""
    import json
    from os.path import exists, isfile
    from os import remove, makedirs
    from glob import glob

    import numpy as np
    from gpaw import GPAW
    from gpaw.mpi import world
    from asr.utils.berryphase import get_polarization_phase

    from ase.parallel import paropen
    from ase.units import Bohr
    from ase.io import jsonio

    from asr.collect import chdir

    if folder is None:
        folder = 'data-borncharges'

    if world.rank == 0:
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    world.barrier()

    with chdir(folder):
        calc = GPAW('../gs.gpw', txt=None)
        params = calc.parameters
        atoms = calc.atoms
        cell_cv = atoms.get_cell() / Bohr
        vol = abs(np.linalg.det(cell_cv))
        sym_a = atoms.get_chemical_symbols()

        pos_av = atoms.get_positions().copy()
        atoms.set_positions(pos_av)
        Z_avv = []
        P_asvv = []

        if world.rank == 0:
            print('Atomnum Atom Direction Displacement')
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
                    berryname = prefix + '-berryphases.json'
                    if not exists(name) and not exists(berryname):
                        calc = get_wavefunctions(atoms, name, params,
                                                 density=kptdensity)
                    try:
                        phase_c = get_polarization_phase(name)
                    except ValueError:
                        calc = get_wavefunctions(atoms, name, params,
                                                 density=kptdensity)
                        phase_c = get_polarization_phase(name)

                    phase_scv[s, :, v] = phase_c

                    if exists(berryname):  # Calculation done?
                        if world.rank == 0:
                            # Remove gpw file
                            if isfile(name):
                                remove(name)

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

        filename = 'borncharges-{}.json'.format(displacement)

        with paropen(filename, 'w') as fd:
            json.dump(jsonio.encode(data), fd)

        world.barrier()
        if world.rank == 0:
            files = glob('born-*.gpw')
            for f in files:
                if isfile(f):
                    remove(f)


def polvsatom(row, *filenames):
    import numpy as np
    if 'borndata' not in row.data:
        return

    from matplotlib import pyplot as plt
    borndata = row.data['borndata']
    deltas = borndata[0]
    P_davv = borndata[1]

    for a, P_dvv in enumerate(P_davv.transpose(1, 0, 2, 3)):
        fname = 'polvsatom{}.png'.format(a)
        for fname2 in filenames:
            if fname in fname2:
                break
        else:
            continue

        Pm_vv = np.mean(P_dvv, axis=0)
        P_dvv -= Pm_vv
        plt.plot(deltas, P_dvv[:, 0, 0], '-o', label='xx')
        plt.plot(deltas, P_dvv[:, 1, 1], '-o', label='yy')
        plt.plot(deltas, P_dvv[:, 2, 2], '-o', label='zz')
        plt.xlabel('Displacement (Å)')
        plt.ylabel('Pol')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname2)
        plt.close()


def webpanel(row, key_descriptions):
    from asr.utils.custom import fig
    polfilenames = []
    if 'Z_avv' in row.data:
        def matrixtable(M, digits=2):
            table = M.tolist()
            shape = M.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = table[i][j]
                    table[i][j] = '{:.{}f}'.format(value, digits)
            return table

        columns = [[], []]
        for a, Z_vv in enumerate(row.data.Z_avv):
            Zdata = matrixtable(Z_vv)

            Ztable = dict(
                header=[str(a), row.symbols[a], ''],
                type='table',
                rows=Zdata)

            columns[0].append(Ztable)
            polname = 'polvsatom{}.png'.format(a)
            columns[1].append(fig(polname))
            polfilenames.append(polname)
        panel = ('Born charges', columns)
    else:
        panel = []
    things = ()
    return panel, things


def collect_data(atoms):
    import json
    import os.path as op
    import numpy as np
    from ase.io import jsonio

    kvp = {}
    data = {}
    key_descriptions = {}
    delta = 0.01
    P_davv = []
    fname = 'data-borncharges/borncharges-{}.json'.format(delta)
    if not op.isfile(fname):
        return {}, {}, {}

    with open(fname) as fd:
        dct = jsonio.decode(json.load(fd))

    P_davv.append(dct['P_asvv'][:, 0])
    P_davv.append(dct['P_asvv'][:, 1])
    data['Z_avv'] = -dct['Z_avv']

    P_davv = np.array(P_davv)
    data['borndata'] = [[-0.01, 0.01], P_davv]

    return kvp, key_descriptions, data


def print_results(filename='data-borncharges/borncharges-0.01.json'):
    import numpy as np
    import json
    from ase.io import jsonio
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    import os.path as op
    if not op.isfile(filename):
        return

    with open(filename) as fd:
        dct = jsonio.decode(json.load(fd))
    title = """
    BORNCHARGES
    ===========
    """
    print(title)
    print(-dct['Z_avv'])


if __name__ == '__main__':
    main.cli()
