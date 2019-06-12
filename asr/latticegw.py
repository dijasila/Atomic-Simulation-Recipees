from asr.utils import command, option


@command('asr.latticegw')
@option('--eta', help='Broadening parameter', default=0.001,
        type=float)
@option('--qcut', help='Cutoff for q-integration', default=0.5)
@option('--microvolume/--no-microvolume', help='Use microvolume integration',
        default=True)
def main(eta, qcut, microvolume):
    """Calculate GW Lattice contribution. """
    import numpy as np
    from ase.units import Hartree
    import json
    from ase.io import jsonio
    from os.path import exists
    from ase.parallel import paropen
    from ase.dft.kpoints import monkhorst_pack
    from gpaw import GPAW
    from gpaw.response.df import DielectricFunction
    from gpaw.response.pair import PairDensity
    from gpaw.wavefunctions.pw import PWDescriptor
    from gpaw.mpi import world
    from gpaw.kpt_descriptor import KPointDescriptor
    from asr.phonons import analyse
    
    print('Calculating GW lattice contribution')

    calc = GPAW('gs.gpw', txt=None)
    # Vibrational frequency (phonon) mode
    print('Loading phonons')
    omega_kl, u_klav, q_qc = analyse(q_qc=[[0, 0, 0]], modes=True)
    u_klav *= 1 / np.sqrt(1822.88)
    print('     Done.')

    # Born Charge Calculation
    print('Extracting effective born charges')
    filename = 'data-borncharges/borncharges-0.01.json'

    with paropen(filename, 'r') as fd:
        data = jsonio.decode(json.load(fd))
    Z_avv = data['Z_avv']
    u_lav = u_klav[0]
    nmodes, natoms, nv = u_lav.shape
    u_lx = u_lav.reshape(-1, natoms * 3)
    Z_xv = Z_avv.reshape(-1, 3)

    Z_lv = np.dot(u_lx, Z_xv)
    Z2_lvv = []
    for Z_v in Z_lv:
        Z2_vv = np.outer(Z_v, Z_v)
        Z2_lvv.append(Z2_vv)
    
    ind = np.argmax(np.abs(Z2_lvv))
    ind = np.unravel_index(ind, (nmodes, 3, 3))
    mode = ind[0]
    Z2_vv = Z2_lvv[mode]
    print('     Done.')

    if not exists('gwlatgs.gpw'):
        print('Performing new groundstate calculation')
        calc = GPAW('gs.gpw',
                    fixdensity=True,
                    kpts={'density': 24.0,
                          'even': True,
                          'gamma': True},
                    nbands=-10,
                    convergence={'bands': -5},
                    txt='gwlatgs.txt')

        calc.get_potential_energy()
        calc.write('gwlatgs.gpw', mode='all')
        print('     Done.')
    else:
        calc = GPAW('gwlatgs.gpw', txt=None)

    # Electronic dielectric constant in infrared
    print('Calculating infrared regime '
          'electronic dielectric constant (epsilon_inf)')
    df = DielectricFunction('G0W0/gwgs.gpw', txt='eps_inf.txt', name='chi0')
    epsmac = df.get_macroscopic_dielectric_constant()[1]
    print('     Done.')

    # Pair-densities
    print('Preparing for calculation of pair-densities')
    # The aim here is to calculate the pair-densities
    ecut = 10
    pair = PairDensity('gwlatgs.gpw', ecut)
    nocc = pair.nocc1

    # Calculation loop for n-bands and k-points
    ikpts = calc.wfs.kd.ibzk_kc
    nikpts = len(ikpts)
    N_c = calc.wfs.kd.N_c

    # various variables (bands and spin)
    s = 0
    nall = nocc + 5
    n_n = np.arange(0, nall)
    m1 = 0
    m2 = nocc
    m_m = np.arange(m1, m2)
    
    # constants for lattice correction term
    volume = pair.vol
    eta = eta / Hartree
    eps = epsmac
    ZBM = (volume * eps * (0.00299**2 - 0.00139**2) / (4 * np.pi))**(1 / 2)
    prefactor = (((4 * np.pi * ZBM) / eps)**2 * 1 / volume)
    freqTO = omega_kl[0, mode] / Hartree
    freqLO = (freqTO**2 + 4 * np.pi * ZBM**2 / (eps * volume))**(1 / 2)
    sigmalat_temp_nk = np.zeros([nall, nikpts], dtype=complex)
    
    # q-dependency
    offset_c = 0.5 * ((N_c + 1) % 2) / N_c
    bzq_qc = monkhorst_pack(N_c) + offset_c
    bzq_qv = np.dot(bzq_qc, calc.wfs.gd.icell_cv) * 2 * np.pi

    qabs_q = np.sum(bzq_qv**2, axis=1)
    qcut = qcut / Hartree
    mask_q = qabs_q / 2 < qcut
    bzq_qc = bzq_qc[mask_q]
    nqtot = len(mask_q)
    nq = len(bzq_qc)
 
    mybzq_qc = bzq_qc[world.rank::world.size]
    myiqs = np.arange(nq)[world.rank::world.size]
    
    print('total number of q:', nqtot)
    print('number of included q:', nq)

    if microvolume:
        prefactor *= nqtot / (2 * np.pi)**3 * volume
    
    B_cv = calc.wfs.gd.icell_cv * 2 * np.pi
    E_cv = B_cv / ((np.sum(B_cv**2, 1))**(1 / 2))[:, None]
    for iq, q_c in zip(myiqs, mybzq_qc):
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(ecut, calc.wfs.gd, complex, qd)
        if microvolume:
            dq_cc = np.eye(3) / N_c[:, None]
            dq_c = ((np.dot(dq_cc, B_cv)**2).sum(1) ** 0.5)
            qd1 = KPointDescriptor([q_c + dq_cc[0]])
            qd2 = KPointDescriptor([q_c + dq_cc[1]])
            qd3 = KPointDescriptor([q_c + dq_cc[2]])
            pd1 = PWDescriptor(ecut, calc.wfs.gd, complex, qd1)
            pd2 = PWDescriptor(ecut, calc.wfs.gd, complex, qd2)
            pd3 = PWDescriptor(ecut, calc.wfs.gd, complex, qd3)

        Q_aGii = pair.initialize_paw_corrections(pd)
        q_v = np.dot(q_c, pd.gd.icell_cv) * 2 * np.pi
        q2abs = np.sum(q_v**2)

        for k in np.arange(0, nikpts):
            # set k-point value
            k_c = ikpts[k]
            # K-point pair (k+-q)
            kptpair = pair.get_kpoint_pair(pd, s, k_c, 0, nall, m1, m2)
            deps0_nm = kptpair.get_transition_energies(n_n, m_m)
            if microvolume:
                kptpair1 = pair.get_kpoint_pair(pd1, s, k_c, 0, nall, m1, m2)
                kptpair2 = pair.get_kpoint_pair(pd2, s, k_c, 0, nall, m1, m2)
                kptpair3 = pair.get_kpoint_pair(pd3, s, k_c, 0, nall, m1, m2)

            if microvolume:
                deps1_nm = kptpair1.get_transition_energies(n_n, m_m)
                deps2_nm = kptpair2.get_transition_energies(n_n, m_m)
                deps3_nm = kptpair3.get_transition_energies(n_n, m_m)
                de_nm = np.array([deps1_nm, deps2_nm, deps3_nm]) - deps0_nm
                v_nmc = de_nm.transpose(1, 2, 0) / dq_c
                v_nmv = np.dot(v_nmc, np.linalg.inv(E_cv).T)

            # -- Pair-Densities -- #
            pairrho_nmG = pair.get_pair_density(pd, kptpair, n_n, m_m,
                                                Q_aGii=Q_aGii,
                                                extend_head=True)
            if np.allclose(q_c, 0.0):
                pairrho2_nm = np.sum(np.abs(pairrho_nmG[:, :, 0:3])**2,
                                     axis=-1) / (3 * volume * nqtot)
                pairrho2_nm[m_m, m_m] = 1 / (4 * np.pi**2) * \
                    ((48 * np.pi**2) / (volume * nqtot))**(1 / 3)
            else:
                pairrho2_nm = (np.abs(pairrho_nmG[:, :, 0])**2 /
                               (volume * nqtot * q2abs))

            for n in n_n:
                pairrho2_m = pairrho2_nm[n]
                deps_m = deps0_nm[n]
                
                # -- Remove degenerate states -- #
                if n < nocc and np.allclose(q_c, 0.0):
                    m = np.where(np.isclose(deps_m, 0.0))[0]
                    for i in np.arange(0, len(m)):
                        if n != m[i]:
                            pairrho2_m[m[i]] = 0.0

                if microvolume:
                    # -- Lattice correction term -- #
                    v_mv = v_nmv[n]
                    v_m = np.linalg.norm(v_mv, axis=1)
                    dq_cv = np.dot(dq_cc, B_cv)
                    qz = np.linalg.norm(dq_cv, axis=1).sum() / 3
                    qrvol = abs(np.linalg.det(dq_cv))
                    qr = np.sqrt(qrvol / (qz * np.pi))

                    muVol_m = (np.pi * qr**2 / (freqLO * v_m) *
                               (np.arctanh((deps_m - v_m * qz - 1j * eta) /
                                           freqLO) -
                                np.arctanh((deps_m - 1j * eta) / freqLO)))

                    corr = np.sum(pairrho2_m * muVol_m)
                else:
                    corr = np.sum(pairrho2_m /
                                  ((deps_m - 1j * eta)**2 - freqTO**2 -
                                   ((4 * np.pi * ZBM**2) / (volume * eps))))
                sigmalat_temp_nk[n, k] += corr

    world.sum(sigmalat_temp_nk)
    sigmalat_nk = prefactor * sigmalat_temp_nk
    data = {'sigmalat_nk': sigmalat_nk}
    return data


def plot():
    import matplotlib
    matplotlib.use('tkagg')
    from matplotlib import pyplot as plt
    import numpy as np
    from gpaw import GPAW
    from asr.utils import read_json
    from ase.units import Hartree

    data = read_json('results_latticegw.json')
    sigmalat_nk = data['sigmalat_nk']
    eval_k = sigmalat_nk[0, :] * Hartree
    calc = GPAW('gwlatgs.gpw', txt=None)
    icell_cv = calc.wfs.gd.icell_cv
    N_c = calc.wfs.kd.N_c
    kd = calc.wfs.kd
    bz2ibz_k = kd.bz2ibz_k

    correction_k = np.zeros(N_c, complex).ravel()

    for k, ik in enumerate(bz2ibz_k):
        correction_k[k] = eval_k[ik]

    kpts_kv = 2 * np.pi * np.dot(calc.wfs.kd.bzk_kc, icell_cv)

    plt.figure()
    correction_kkk = correction_k.reshape(N_c)
    kpts_kkkv = kpts_kv.reshape(list(N_c) + [3])

    ind = N_c[0] // 2 - 1
    
    slc_kk = correction_kkk[ind, :, :]
    slk_kkv = kpts_kkkv[ind, :, :, :]
    plt.scatter(slk_kkv[:, :, 0], slk_kkv[:, :, 1], s=0.5, c='black', zorder=2)
    plt.pcolormesh(slk_kkv[:, :, 0], slk_kkv[:, :, 1], slc_kk.real)
    plt.colorbar()

    plt.figure()
    plt.scatter(slk_kkv[:, :, 0], slk_kkv[:, :, 1], s=0.5, c='black', zorder=2)
    plt.contourf(slk_kkv[:, :, 0], slk_kkv[:, :, 1], slc_kk.real, levels=40)
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main()
