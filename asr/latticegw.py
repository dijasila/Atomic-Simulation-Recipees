from asr.utils import command, option


@command('asr.latticegw')
@option('--ecut', help='Plane wave cut off', default=10,
        type=float)
def main(ecut):
    """Calculate GW Lattice contribution. """
    import numpy as np
    from ase.units import Hartree
    import json
    from ase.io import jsonio
    import os
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
    
    if not exists('GW+lat-bs'):
        os.makedirs('GW+lat-bs')
    
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

    if not exists('GW+lat-bs/gwlatgs.gpw'):
        print('Performing new groundstate calculation')
        calc = GPAW('gs.gpw',
                    fixdensity=True,
                    kpts={'density': 12.0,
                          'even': True,
                          'gamma': True},
                    nbands=-10,
                    convergence={'bands': -5},
                    txt='GW+lat-bs/gwlatgs.txt')

        calc.get_potential_energy()
        calc.write('GW+lat-bs/gwlatgs.gpw', mode='all')
        print('     Done.')
    else:
        calc = GPAW('GW+lat-bs/gwlatgs.gpw', txt=None)

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
    pair = PairDensity('GW+lat-bs/gwlatgs.gpw', ecut)
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
    eta = 0.001 / Hartree
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
    qecut = 0.5 / Hartree
    mask_q = qabs_q / 2 < qecut
    bzq_qc = bzq_qc[mask_q]
    nqtot = len(mask_q)
    nq = len(bzq_qc)
 
    mybzq_qc = bzq_qc[world.rank::world.size]
    myiqs = np.arange(nq)[world.rank::world.size]
    
    print('total number of q:', nqtot)
    print('number of included q:', nq)
    
    prefactor *= nqtot / (2 * np.pi)**3 * volume
    
    qabs_qecut = np.zeros(nq)
    B_cv = calc.wfs.gd.icell_cv * 2 * np.pi
    for iq, q_c in zip(myiqs, mybzq_qc):
        dq_cc = np.eye(3) / N_c[:, None]
        dq_c = ((np.dot(dq_cc, B_cv)**2).sum(1) ** 0.5)

        qd = KPointDescriptor([q_c])
        qd1 = KPointDescriptor([q_c + dq_cc[0]])
        qd2 = KPointDescriptor([q_c + dq_cc[1]])
        qd3 = KPointDescriptor([q_c + dq_cc[2]])
        pd = PWDescriptor(ecut, calc.wfs.gd, complex, qd)
        pd1 = PWDescriptor(ecut, calc.wfs.gd, complex, qd1)
        pd2 = PWDescriptor(ecut, calc.wfs.gd, complex, qd2)
        pd3 = PWDescriptor(ecut, calc.wfs.gd, complex, qd3)

        Q_aGii = pair.initialize_paw_corrections(pd)
        q_v = np.dot(q_c, pd.gd.icell_cv) * 2 * np.pi
        q2abs = np.sum(q_v**2)
        B_cv = pd.gd.icell_cv * 2 * np.pi
        E_cv = B_cv / ((np.sum(B_cv**2, 1))**(1 / 2))[:, None]

        qabs_qecut[iq] = q2abs**(1 / 2)
        
        for k in np.arange(0, nikpts):
            # set k-point value
            k_c = ikpts[k]
            # K-point pair (k+-q)
            kptpair = pair.get_kpoint_pair(pd, s, k_c, 0, nall, m1, m2)
            kptpair1 = pair.get_kpoint_pair(pd1, s, k_c, 0, nall, m1, m2)
            kptpair2 = pair.get_kpoint_pair(pd2, s, k_c, 0, nall, m1, m2)
            kptpair3 = pair.get_kpoint_pair(pd3, s, k_c, 0, nall, m1, m2)
            # -- Kohn-Sham energy difference (e_n - e_m) -- #
            deps0_nm = kptpair.get_transition_energies(n_n, m_m)
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
                sigmalat_temp_nk[n, k] += corr

                corr = np.sum(pairrho2_m /
                              ((deps_m - 1j * eta)**2 - freqTO**2 -
                               ((4 * np.pi * ZBM**2) / (volume * eps))))
    world.sum(sigmalat_temp_nk)
    sigmalat_nk = prefactor * sigmalat_temp_nk
    data = {'sigmalat_nk': sigmalat_nk}
    return sigmalat_nk


if __name__ == '__main__':
    main()
