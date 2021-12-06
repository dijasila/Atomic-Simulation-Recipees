"""Calculate the lattice polarizability GW self-energy contribution."""

import numpy as np
from ase.dft.kpoints import monkhorst_pack
from ase.units import Hartree
from gpaw import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import world
from gpaw.response.df import DielectricFunction
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor

import asr
from asr.c2db.borncharges import main as calculate_borncharges
from asr.c2db.phonons import main as calculate_phonons
from asr.core.utils import file_barrier


@asr.instruction("asr.latticegw")
@asr.atomsopt()
@asr.calcopt()
@asr.option("-n", help="Supercell size", type=int)
@asr.option(
    "--mingo/--no-mingo",
    is_flag=True,
    help="Perform Mingo correction of force constant matrix",
)
@asr.option("--eta", help="Broadening parameter", default=0.01, type=float)
@asr.option("--qcut", help="Cutoff for q-integration", default=2, type=float)
@asr.option(
    "--microvolume/--no-microvolume", help="Use microvolume integration", default=True
)
@asr.option(
    "--maxband",
    default=None,
    type=int,
    help="Maximum band index to calculate correction of",
)
@asr.option(
    "--kptdensity",
    default=12,
    type=float,
    help="K Point density for ground state calculation",
)
@asr.calcopt(aliases=["--borncalculator"])
@asr.option(
    "--displacement",
    help="Born charge displacement.",
)
def main(
    atoms,
    calculator,
    n,
    mingo,
    eta,
    qcut,
    maxband,
    kptdensity,
    borncalculator,
    displacement,
):
    """Calculate GW Lattice contribution."""

    phononresult = calculate_phonons(atoms, calculator, n, mingo)
    omega_kl, u_klav = phononresult.omega_kl, phononresult.modes_kl
    u_klav *= 1 / np.sqrt(1822.88)
    borncharges = calculate_borncharges(
        atoms=atoms,
        calculator=borncalculator,
        displacement=displacement,
    )
    Z_avv = borncharges.Z_avv
    u_lav = u_klav[0]
    nmodes, natoms, _ = u_lav.shape
    u_lx = u_lav.reshape(-1, natoms * 3)
    Z_xv = Z_avv.reshape(-1, 3)

    Z_lv = np.dot(u_lx, Z_xv)
    Z2_lvv = []
    for Z_v in Z_lv:
        Z2_vv = np.outer(Z_v, Z_v)
        Z2_lvv.append(Z2_vv)

    ind = np.argmax(np.abs(Z2_lvv))
    ind = np.unravel_index(ind, (nmodes, 3, 3))
    mode: int = ind[0]
    Z2_vv = Z2_lvv[mode]

    calc = GPAW(
        "gs.gpw",
        fixdensity=True,
        kpts={"density": kptdensity, "even": True, "gamma": True},
        nbands=-10,
        convergence={"bands": -5},
        txt="gwlatgs.txt",
    )

    calc.get_potential_energy()

    with file_barrier("gwlatgs.gpw"):
        calc.write("gwlatgs.gpw", mode="all")

    # Electronic dielectric constant in infrared
    df = DielectricFunction("G0W0/gwgs.gpw", txt="eps_inf.txt", name="chi0")
    epsmac = df.get_macroscopic_dielectric_constant()[1]
    ecut = 10
    pair = PairDensity("gwlatgs.gpw", ecut)
    nocc = pair.nocc1
    ikpts = calc.wfs.kd.ibzk_kc
    nikpts = len(ikpts)
    N_c = calc.wfs.kd.N_c
    s = 0
    nall = maxband or nocc + 5
    n_n = np.arange(0, nall)
    m1 = 0
    if maxband and maxband < nocc:
        m2 = maxband
    else:
        m2 = nocc

    m_m = np.arange(m1, m2)
    volume = pair.vol
    eta = eta / Hartree
    eps = epsmac
    zbm = (volume * eps * (0.00299 ** 2 - 0.00139 ** 2) / (4 * np.pi)) ** (1 / 2)

    freq_to = omega_kl[0, mode] / Hartree
    freq_lo = (freq_to ** 2 + 4 * np.pi * zbm ** 2 / (eps * volume)) ** (1 / 2)
    freq_lo2 = freq_lo ** 2

    # q-dependency
    offset_c = 0.5 * ((N_c + 1) % 2) / N_c
    bzq_qc = monkhorst_pack(N_c) + offset_c
    bzq_qv = np.dot(bzq_qc, calc.wfs.gd.icell_cv) * 2 * np.pi

    qabs_q = np.sum(bzq_qv ** 2, axis=1)
    qcut = qcut / Hartree
    mask_q = qabs_q / 2 < qcut
    bzq_qc = bzq_qc[mask_q]
    nqtot = len(mask_q)

    mybzq_qc = bzq_qc[world.rank :: world.size]

    prefactor = -(((4 * np.pi * zbm) / eps) ** 2) / (volume ** 2 * nqtot)

    sigmalat_nk = np.zeros([nall, nikpts], dtype=complex)
    iq = 0
    for q_c in mybzq_qc:
        print(iq)
        iq += 1
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(ecut, calc.wfs.gd, complex, qd)

        Q_aGii = pair.initialize_paw_corrections(pd)
        q_v = np.dot(q_c, pd.gd.icell_cv) * 2 * np.pi
        q2abs = np.sum(q_v ** 2)

        for k in np.arange(0, nikpts):
            k_c = ikpts[k]
            kptpair = pair.get_kpoint_pair(pd, s, k_c, 0, nall, m1, m2)
            deps0_nm = kptpair.get_transition_energies(n_n, m_m)

            # -- Pair-Densities -- #
            pairrho_nmG = pair.get_pair_density(
                pd, kptpair, n_n, m_m, Q_aGii=Q_aGii, extend_head=True
            )

            if np.allclose(q_c, 0.0):
                pairrho2_nm = np.sum(np.abs(pairrho_nmG[:, :, 0:3]) ** 2, axis=-1) / 3
                pairrho2_nm[m_m, m_m] = (
                    1
                    / (4 * np.pi ** 2)
                    * ((48 * np.pi ** 2) / (volume * nqtot)) ** (1 / 3)
                )
                pairrho2_nm *= volume * nqtot
            else:
                pairrho2_nm = np.abs(pairrho_nmG[:, :, 0]) ** 2 / q2abs

            deps0_nm = deps0_nm - 1j * eta
            corr_n = np.sum(pairrho2_nm / (deps0_nm ** 2 - freq_lo2), axis=1)
            sigmalat_nk[:, k] += corr_n

    world.sum(sigmalat_nk)
    sigmalat_nk *= prefactor
    return sigmalat_nk
