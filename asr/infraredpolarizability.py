"""Infrared polarizability."""
import numpy as np
from asr.core import command, option, read_json
from asr.paneldata import InfraredResult


@command(
    "asr.infraredpolarizability",
    dependencies=["asr.phonons", "asr.borncharges", "asr.polarizability"],
    requires=[
        "structure.json",
        "results-asr.phonons.json",
        "results-asr.borncharges.json",
        "results-asr.polarizability.json",
    ],
    returns=InfraredResult,
)
@option("--nfreq", help="Number of frequency points", type=int)
@option("--eta", help="Relaxation rate", type=float)
def main(nfreq: int = 300, eta: float = 1e-2) -> InfraredResult:
    from ase.io import read

    # Get relevant atomic structure
    atoms = read("structure.json")

    # Get phonons
    phresults = read_json("results-asr.phonons.json")
    u_ql = phresults["modes_kl"]
    q_qc = phresults["q_qc"]
    omega_ql = phresults["omega_kl"]

    iq_q = np.argwhere((np.abs(q_qc) < 1e-10).all(axis=1))

    assert len(iq_q), "Calculated phonons do not contain Gamma point."

    iq = iq_q[0][0]

    m_a = atoms.get_masses()
    m_inv_x = np.repeat(m_a ** -0.5, 3)
    freqs_l, modes_liv = omega_ql[iq], u_ql[iq]
    modes_xl = modes_liv.reshape(len(freqs_l), -1).T
    modes_xl *= 1 / m_inv_x[:, np.newaxis]

    # Make frequency grid
    fmin = 0
    fmax = omega_ql[0].max() * 3  # Factor of 3 should be enough
    omega_w = np.linspace(fmin, fmax, nfreq)

    # Read born charges
    borndct = read_json("results-asr.borncharges.json")

    # Get other relevant quantities
    m_a = atoms.get_masses()
    cell_cv = atoms.get_cell()
    Z_avv = borndct["Z_avv"]

    # Get phonon polarizability
    alpha_wvv = get_phonon_pol(omega_w, Z_avv, freqs_l, modes_xl, m_a, cell_cv, eta)

    # Normalize according to dimensionality
    pbc_c = atoms.pbc
    if pbc_c.all():
        norm = 1
    else:
        norm = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
    alpha_wvv *= norm

    alphax_lat = alpha_wvv[0, 0, 0].real
    alphay_lat = alpha_wvv[0, 1, 1].real
    alphaz_lat = alpha_wvv[0, 2, 2].real

    elecdict = read_json("results-asr.polarizability.json")
    alphax_el = elecdict["alphax_el"]
    alphay_el = elecdict["alphay_el"]
    alphaz_el = elecdict["alphaz_el"]

    results = {
        "alpha_wvv": alpha_wvv,
        "omega_w": omega_w,
        "alphax_lat": alphax_lat,
        "alphay_lat": alphay_lat,
        "alphaz_lat": alphaz_lat,
        "alphax": alphax_lat + alphax_el,
        "alphay": alphay_lat + alphay_el,
        "alphaz": alphaz_lat + alphaz_el,
    }

    return results


def get_phonon_pol(omega_w, Z_avv, freqs_l, modes_xl, m_a, cell_cv, eta):
    from ase.units import Hartree, Bohr

    Z_vx = Z_avv.swapaxes(0, 1).reshape((3, -1))
    f2_w, D_xw = (freqs_l / Hartree) ** 2, modes_xl

    vol = abs(np.linalg.det(cell_cv)) / Bohr ** 3
    omega_w = omega_w / Hartree
    eta = eta / Hartree
    me = 1822.888
    m_a = m_a * me
    alpha_wvv = np.zeros((len(omega_w), 3, 3), dtype=complex)
    m_x = np.repeat(m_a, 3) ** 0.5
    eta = eta

    for f2, D_x in zip(f2_w, D_xw.T):
        # Neglect acoustic modes
        if f2 < (1e-3 / Hartree) ** 2:
            continue
        DM_x = D_x / m_x
        Z_v = np.dot(Z_vx, DM_x)
        alpha_wvv += (
            np.outer(Z_v, Z_v)[np.newaxis]
            / ((f2 - omega_w ** 2) - 1j * eta * omega_w)[:, np.newaxis, np.newaxis]
        )

    alpha_wvv /= vol
    return alpha_wvv


if __name__ == "__main__":
    main()
