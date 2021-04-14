"""Infrared polarizability."""
import typing
import asr
from asr.core import (
    command, option, ASRResult, prepare_result, atomsopt,
    Selector, make_migration_generator,
)
from asr.database.browser import (
    fig, table, href, make_panel_description, describe_entry)

import numpy as np
from click import Choice

from ase import Atoms
from asr.phonons import main as phonons
from asr.borncharges import main as borncharges
from asr.polarizability import main as polarizability

panel_description = make_panel_description(
    """The frequency-dependent polarisability in the infrared (IR) frequency regime
calculated from a Lorentz oscillator equation involving the optical Gamma-point
phonons and atomic Born charges. The contribution from electronic interband
transitions is added, but is essentially constant for frequencies much smaller
than the direct band gap.
""",
    articles=[
        href("""M. N. Gjerding et al. Efficient Ab Initio Modeling of Dielectric Screening
        in 2D van der Waals Materials: Including Phonons, Substrates, and Doping,
        J. Phys. Chem. C 124 11609 (2020)""",
             'https://doi.org/10.1021/acs.jpcc.0c01635'),
    ]

)


def webpanel(result, row, key_descriptions):

    opt = table(
        row, "Property", ["alphax_lat", "alphay_lat", "alphaz_lat"], key_descriptions
    )

    panel = {
        "title": describe_entry("Infrared polarizability (RPA)",
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
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # Get electronic polarizability
    infrareddct = row.data.get("results-asr.infraredpolarizability.json")
    omega_w = infrareddct["omega_w"] * 1e3
    alpha_wvv = infrareddct["alpha_wvv"]

    electrondct = row.data.get("results-asr.polarizability.json")
    alphax_w = electrondct["alphax_w"]
    alphay_w = electrondct["alphay_w"]
    alphaz_w = electrondct["alphaz_w"]
    omegatmp_w = electrondct["frequencies"] * 1e3

    # Get max phonon freq
    phonondata = row.data.get("results-asr.phonons.json")
    maxphononfreq = phonondata.get("omega_kl")[0].max() * 1e3
    maxomega = maxphononfreq * 1.5

    atoms = row.toatoms()
    pbc_c = atoms.pbc
    ndim = int(np.sum(pbc_c))

    realphax = interp1d(omegatmp_w, alphax_w.real)
    imalphax = interp1d(omegatmp_w, alphax_w.imag)
    ax_w = (realphax(omega_w) + 1j * imalphax(omega_w)
            + alpha_wvv[:, 0, 0])
    realphay = interp1d(omegatmp_w, alphay_w.real)
    imalphay = interp1d(omegatmp_w, alphay_w.imag)
    ay_w = (realphay(omega_w) + 1j * imalphay(omega_w)
            + alpha_wvv[:, 1, 1])
    realphaz = interp1d(omegatmp_w, alphaz_w.real)
    imalphaz = interp1d(omegatmp_w, alphaz_w.imag)
    az_w = (realphaz(omega_w) + 1j * imalphaz(omega_w)
            + alpha_wvv[:, 2, 2])

    if ndim == 3:
        epsx_w = 1 + 4 * np.pi * ax_w
        epsy_w = 1 + 4 * np.pi * ay_w
        epsz_w = 1 + 4 * np.pi * az_w
        plt.figure()
        plt.plot(omega_w, epsx_w.imag, label='imag')
        plt.plot(omega_w, epsx_w.real, label='real')
        ax = plt.gca()
        ax.set_title("x-polarization")
        ax.set_xlabel("Energy [meV]")
        ax.set_ylabel(r"Dielectric function")
        ax.set_xlim(0, maxomega)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fnames[0])

        plt.figure()
        plt.plot(omega_w, epsy_w.imag, label='imag')
        plt.plot(omega_w, epsy_w.real, label='real')
        ax = plt.gca()
        ax.set_title("y-polarization")
        ax.set_xlabel("Energy [meV]")
        ax.set_ylabel(r"Dielectric function")
        ax.set_xlim(0, maxomega)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fnames[1])

        plt.figure()
        plt.plot(omega_w, epsz_w.imag, label='imag')
        plt.plot(omega_w, epsz_w.real, label='real')
        ax = plt.gca()
        ax.set_title("z-polarization")
        ax.set_xlabel("Energy [meV]")
        ax.set_ylabel(r"Dielectric function")
        ax.set_xlim(0, maxomega)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fnames[2])
    elif ndim in [2, 1, 0]:
        if ndim == 2:
            unit = r"$\mathrm{\AA}$"
        elif ndim == 1:
            unit = r"$\mathrm{\AA}^2$"
        elif ndim == 0:
            unit = r"$\mathrm{\AA}^3$"
        plt.figure()
        plt.plot(omega_w, ax_w.imag, label='imag')
        plt.plot(omega_w, ax_w.real, label='real')
        ax = plt.gca()
        ax.set_title("x-polarization")
        ax.set_xlabel("Energy [meV]")
        ax.set_ylabel(rf"Polarizability [{unit}]")
        ax.set_xlim(0, maxomega)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fnames[0])

        plt.figure()
        plt.plot(omega_w, ay_w.imag, label='imag')
        plt.plot(omega_w, ay_w.real, label='real')
        ax = plt.gca()
        ax.set_title("y-polarization")
        ax.set_xlabel("Energy [meV]")
        ax.set_ylabel(rf"Polarizability [{unit}]")
        ax.set_xlim(0, maxomega)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fnames[1])

        plt.figure()
        plt.plot(omega_w, az_w.imag, label='imag')
        plt.plot(omega_w, az_w.real, label='real')
        ax = plt.gca()
        ax.set_title("z-polarization")
        ax.set_xlabel("Energy [meV]")
        ax.set_ylabel(rf"Polarizability [{unit}]")
        ax.set_xlim(0, maxomega)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fnames[2])


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


def prepare_for_resultfile_migration(record):
    """Prepare record for resultfile migration."""
    phononpar = record.parameters.dependency_parameters['asr.phonons:calculate']
    fconverge = phononpar['fconverge']
    del phononpar['fconverge']
    record.parameters.bandfactor = 5
    record.parameters.xc = 'RPA'
    record.parameters.phononcalculator = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'kpts': {'density': 6.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'forces': fconverge},
        'symmetry': {'point_group': False},
        'nbands': '200%',
        'txt': 'phonons.txt',
        'charge': 0,
    }
    record.parameters.borncalculator = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'kpts': {'density': 12.0},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'symmetry': 'off',
        'convergence': {'eigenstates': 1e-11,
                        'density': 1e-7},
        'txt': 'formalpol.txt',
        'charge': 0,
    }
    record.parameters.polarizabilitycalculator = \
        record.parameters.dependency_parameters['asr.gs:calculate']['calculator']
    del record.parameters.dependency_parameters['asr.gs:calculate']['calculator']
    return record


sel = Selector()
sel.version = sel.EQ(-1)
sel.parameters = sel.NOT(
    sel.ANY(
        sel.CONTAINS('bandfactor'),
        sel.CONTAINS('xc'),
        sel.CONTAINS('phononcalculator'),
    )
)
sel.name = sel.EQ('asr.infraredpolarizability:main')


make_migrations = make_migration_generator(
    selector=sel,
    function=prepare_for_resultfile_migration,
    uid='048c99cc09c641c187929ed67d9ffc39',
)


@command(
    "asr.infraredpolarizability",
    migrations=[make_migrations],
)
@atomsopt
@asr.calcopt(aliases=['-b', '--borncalculator'], help='Born calculator.')
@asr.calcopt(aliases=['-p', '--phononcalculator'], help='Phonon calculator.')
@asr.calcopt(aliases=['-a', '--polarizabilitycalculator'],
             help='Polarizability calculator.')
@option("--nfreq", help="Number of frequency points", type=int)
@option("--eta", help="Relaxation rate", type=float)
@option('-n', help='Supercell size', type=int)
@option('--mingo/--no-mingo', is_flag=True,
        help='Perform Mingo correction of force constant matrix')
@option('--displacement', help='Atomic displacement (Å)', type=float)
@option('--kptdensity', help='K-point density',
        type=float)
@option('--ecut', help='Plane wave cutoff',
        type=float)
@option('--xc', help='XC interaction', type=Choice(['RPA', 'ALDA']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def main(
        atoms: Atoms,
        nfreq: int = 300,
        eta: float = 1e-2,
        polarizabilitycalculator: dict = polarizability.defaults.calculator,
        phononcalculator: dict = phonons.defaults.calculator,
        borncalculator: dict = borncharges.defaults.calculator,
        n: int = phonons.defaults.n,
        mingo: bool = phonons.defaults.mingo,
        displacement: float = borncharges.defaults.displacement,
        kptdensity: float = polarizability.defaults.kptdensity,
        ecut: float = polarizability.defaults.ecut,
        xc: float = polarizability.defaults.xc,
        bandfactor: float = polarizability.defaults.bandfactor,
) -> Result:

    # Get phonons
    phresults = phonons(
        atoms=atoms,
        calculator=phononcalculator,
        n=n,
        mingo=mingo,
    )

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

    borndct = borncharges(
        atoms=atoms,
        calculator=borncalculator,
        displacement=displacement,
    )

    # Get other relevant quantities
    m_a = atoms.get_masses()
    cell_cv = atoms.get_cell()
    Z_avv = borndct["Z_avv"]

    # Get phonon polarizability
    alpha_wvv = get_phonon_pol(
        omega_w, Z_avv, freqs_l,
        modes_xl, m_a, cell_cv, eta,
    )

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

    elecdict = polarizability(
        atoms=atoms,
        calculator=polarizabilitycalculator,
        kptdensity=kptdensity,
        ecut=ecut,
        xc=xc,
        bandfactor=bandfactor,
    )
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

    return Result(data=results)


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
