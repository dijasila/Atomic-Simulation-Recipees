import json
from os.path import exists, splitext

import numpy as np
from gpaw import GPAW
from gpaw.mpi import serial_comm, world

from ase.dft.kpoints import get_monkhorst_pack_size_and_offset


def get_overlap(calc, bands, u1_nR, u2_nR, P1_ani, P2_ani, dO_aii, bG_v):
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
