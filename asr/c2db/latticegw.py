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


@asr.instruction("asr.latticegw")
@asr.atomsopt()
@asr.calcopt()
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
def main(
    atoms,
    calculator,
    n,
    mingo,
    eta,
    qcut,
    microvolume,
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

    calc = GPAW(
        "gs.gpw",
        fixdensity=True,
        kpts={"density": kptdensity, "even": True, "gamma": True},
        nbands=-10,
        convergence={"bands": -5},
        txt="gwlatgs.txt",
    )

    calc.get_potential_energy()
    from asr.utils import file_barrier

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
    ZBM = (volume * eps * (0.00299 ** 2 - 0.00139 ** 2) / (4 * np.pi)) ** (1 / 2)

    freqTO = omega_kl[0, mode] / Hartree
    freqLO = (freqTO ** 2 + 4 * np.pi * ZBM ** 2 / (eps * volume)) ** (1 / 2)
    freqLO2 = freqLO ** 2

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

    B_cv = calc.wfs.gd.icell_cv * 2 * np.pi
    dq_c = 1 / N_c * [1, 0, 0]
    dq_v = (B_cv / N_c[:, None])[0]
    dq = np.sum(dq_v ** 2) ** 0.5
    dqvol = (2 * np.pi) ** 3 / volume / nqtot
    qr = np.sqrt(dqvol / (dq * np.pi))

    prefactor = -(((4 * np.pi * ZBM) / eps) ** 2) / (volume ** 2 * nqtot)

    sigmalat_nk = np.zeros([nall, nikpts], dtype=complex)
    iq = 0
    for q_c in mybzq_qc:
        print(iq)
        iq += 1
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(ecut, calc.wfs.gd, complex, qd)
        if microvolume:
            qd1 = KPointDescriptor([q_c + dq_c])
            pd1 = PWDescriptor(ecut, calc.wfs.gd, complex, qd1)

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
            corr_n = np.sum(pairrho2_nm / (deps0_nm ** 2 - freqLO2), axis=1)
            sigmalat_nk[:, k] += corr_n

    world.sum(sigmalat_nk)
    sigmalat_nk *= prefactor
    return sigmalat_nk


def plot():
    import matplotlib

    matplotlib.use("tkagg")
    import numpy as np
    from ase.units import Hartree
    from gpaw import GPAW
    from matplotlib import pyplot as plt

    from asr.utils import read_json

    data = read_json("results_latticegw.json")
    sigmalat_nk = data["sigmalat_nk"]
    eval_k = sigmalat_nk[0, :] * Hartree
    calc = GPAW("gwlatgs.gpw", txt=None)
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
    plt.scatter(slk_kkv[:, :, 0], slk_kkv[:, :, 1], s=0.5, c="black", zorder=2)
    plt.pcolormesh(slk_kkv[:, :, 0], slk_kkv[:, :, 1], slc_kk.real)
    plt.colorbar()

    plt.figure()
    plt.scatter(slk_kkv[:, :, 0], slk_kkv[:, :, 1], s=0.5, c="black", zorder=2)
    plt.contourf(slk_kkv[:, :, 0], slk_kkv[:, :, 1], slc_kk.real, levels=40)
    plt.colorbar()

    plt.show()


def atoms2bandstructure(atoms, eps_skn=None, path=None, points=300):
    import matplotlib

    matplotlib.use("tkagg")
    import numpy as np
    from ase.dft.band_structure import BandStructure
    from ase.dft.kpoints import (
        BandPath,
        bandpath,
        get_monkhorst_pack_size_and_offset,
        get_special_points,
        monkhorst_pack_interpolate,
    )
    from ase.geometry import crystal_structure_from_cell

    cell = atoms.get_cell()
    calc = atoms.calc
    bzkpts = calc.get_bz_k_points()
    ibzkpts = calc.get_ibz_k_points()
    efermi = calc.get_fermi_level()
    nibz = len(ibzkpts)
    nspins = 1 + int(calc.get_spin_polarized())

    if eps_skn is None:
        eps_skn = np.array(
            [
                [calc.get_eigenvalues(kpt=k, spin=s) for k in range(nibz)]
                for s in range(nspins)
            ]
        )

    print("Spins, k-points, bands: {}, {}, {}".format(*eps_skn.shape))
    try:
        size, offset = get_monkhorst_pack_size_and_offset(bzkpts)
    except ValueError:
        path_kpts = ibzkpts
    else:
        print("Interpolating from Monkhorst-Pack grid (size, offset):")
        print(size, offset)
        if path is None:
            cs = crystal_structure_from_cell(cell)
            from ase.dft.kpoints import special_paths

            kptpath = special_paths[cs]
            path = kptpath

        bz2ibz = calc.get_bz_to_ibz_map()

        path_kpts = bandpath(path, atoms.cell, points)[0]
        icell = atoms.get_reciprocal_cell()
        eps = monkhorst_pack_interpolate(
            path_kpts, eps_skn.transpose(1, 0, 2), icell, bz2ibz, size, offset
        )
        eps = eps.transpose(1, 0, 2)

    special_points = get_special_points(cell)
    path = BandPath(atoms.cell, kpts=path_kpts, special_points=special_points)

    return BandStructure(path, eps, reference=efermi)


def plotbs():
    # import numpy as np
    from ase.io import read
    from ase.units import Hartree

    from asr.utils import read_json

    bs = atoms2bandstructure(read("gwlatgs.gpw"))
    ax = bs.plot(emin=-20, emax=20, show=False)

    data = read_json("results_latticegw.json")
    sigmalat_nk = data["sigmalat_nk"]
    eps_skn = sigmalat_nk.T[None].real * Hartree
    nbands = sigmalat_nk.shape[0]
    bs2 = atoms2bandstructure(read("gwlatgs.gpw"), eps_skn=eps_skn)
    bs2.energies += bs.energies[:, :, :nbands]
    bs2.plot(emin=-20, emax=20, ax=ax, colors="r")


dependencies = ["asr.gs", "asr.phonons"]
creates = ["results_latticegw.json"]

if __name__ == "__main__":
    main()
