"""Infrared polarizability."""
import typing
from asr.core import command, option, read_json, ASRResult, prepare_result
from asr.database.browser import (
    fig, table, href, make_panel_description, describe_entry)

import numpy as np

reference = """\
M. N. Gjerding et al. Efficient Ab Initio Modeling of Dielectric Screening
in 2D van der Waals Materials: Including Phonons, Substrates, and Doping,
J. Phys. Chem. C 124 11609 (2020)"""

panel_description = make_panel_description(
    """The frequency-dependent polarisability in the infrared (IR) frequency regime
calculated from a Lorentz oscillator equation involving the optical Gamma-point
phonons and atomic Born charges. The contribution from electronic interband
transitions is added, but is essentially constant for frequencies much smaller
than the direct band gap.
""",
    articles=[
        href(reference, 'https://doi.org/10.1021/acs.jpcc.0c01635'),
    ]

)


def webpanel(result, row, key_descriptions):
    explanation = 'Static lattice polarizability along the'
    alphax_lat = describe_entry('alphax_lat', description=explanation + " x-direction")
    alphay_lat = describe_entry('alphay_lat', description=explanation + " y-direction")
    alphaz_lat = describe_entry('alphaz_lat', description=explanation + " z-direction")

    explanation = 'Total static polarizability along the'
    alphax = describe_entry('alphax', description=explanation + " x-direction")
    alphay = describe_entry('alphay', description=explanation + " y-direction")
    alphaz = describe_entry('alphaz', description=explanation + " z-direction")

    opt = table(
        row, "Property", [alphax_lat, alphay_lat, alphaz_lat, alphax,
                          alphay, alphaz], key_descriptions
    )

    panel = {
        "title": describe_entry("Infrared polarizability",
                                panel_description),
        "columns": [[fig("infrax.png"), fig("infraz.png")], [fig("infray.png"), opt]],
        "plot_descriptions": [
            {
                "function": create_plot,
                "filenames": ["infrax.png", "infray.png", "infraz.png"],
            }
        ],
        "sort": 21,
    }

    return [panel]


def create_plot(row, *fnames):
    infrareddct = row.data['results-asr.infraredpolarizability.json']
    electrondct = row.data['results-asr.polarizability.json']
    phonondata = row.data['results-asr.phonons.json']
    maxphononfreq = phonondata['omega_kl'][0].max() * 1e3

    assert len(fnames) == 3
    for v, (axisname, fname) in enumerate(zip('xyz', fnames)):
        alpha_w = electrondct[f'alpha{axisname}_w']

        create_plot_simple(
            ndim=sum(row.toatoms().pbc),
            maxomega=maxphononfreq * 1.5,
            omega_w=infrareddct["omega_w"] * 1e3,
            alpha_w=alpha_w,
            alphavv_w=infrareddct["alpha_wvv"][:, v, v],
            omegatmp_w=electrondct["frequencies"] * 1e3,
            axisname=axisname,
            fname=fname)


def create_plot_simple(*, ndim, omega_w, fname, maxomega, alpha_w,
                       alphavv_w, axisname,
                       omegatmp_w):
    from scipy.interpolate import interp1d

    re_alpha = interp1d(omegatmp_w, alpha_w.real)
    im_alpha = interp1d(omegatmp_w, alpha_w.imag)
    a_w = (re_alpha(omega_w) + 1j * im_alpha(omega_w) + alphavv_w)

    if ndim == 3:
        ylabel = r'Dielectric function'
        yvalues = 1 + 4 * np.pi * a_w
    else:
        power_txt = {2: '', 1: '^2', 0: '^3'}[ndim]
        unit = rf"$\mathrm{{\AA}}{power_txt}$"
        ylabel = rf'Polarizability [{unit}]'
        yvalues = a_w

    return mkplot(yvalues, axisname, fname, maxomega, omega_w, ylabel)


def mkplot(a_w, axisname, fname, maxomega, omega_w, ylabel):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(omega_w, a_w.real, c='C1', label='real')
    ax.plot(omega_w, a_w.imag, c='C0', label='imag')
    ax.set_title(f'Polarization: {axisname}')
    ax.set_xlabel('Energy [meV]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, maxomega)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


@prepare_result
class Result(ASRResult):

    alpha_wvv: typing.List[typing.List[typing.List[complex]]]
    omega_w: typing.List[float]
    alphax_lat: complex
    alphay_lat: complex
    alphaz_lat: complex
    alphax: complex
    alphay: complex
    alphaz: complex

    key_descriptions = {
        "alpha_wvv": "Lattice polarizability.",
        "omega_w": "Frequency grid [eV].",
        "alphax_lat": "Lattice polarizability at omega=0 (x-direction).",
        "alphay_lat": "Lattice polarizability at omega=0 (y-direction).",
        "alphaz_lat": "Lattice polarizability at omega=0 (z-direction).",
        "alphax": "Lattice+electronic polarizability at omega=0 (x-direction).",
        "alphay": "Lattice+electronic polarizability at omega=0 (y-direction).",
        "alphaz": "Lattice+electronic polarizability at omega=0 (z-direction).",
    }

    formats = {"ase_webpanel": webpanel}


@command(
    "asr.infraredpolarizability",
    dependencies=["asr.phonons", "asr.borncharges", "asr.polarizability"],
    requires=[
        "structure.json",
        "results-asr.phonons.json",
        "results-asr.borncharges.json",
        "results-asr.polarizability.json",
    ],
    returns=Result,
)
@option("--nfreq", help="Number of frequency points", type=int)
@option("--eta", help="Relaxation rate", type=float)
def main(nfreq: int = 300, eta: float = 1e-2) -> Result:
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
