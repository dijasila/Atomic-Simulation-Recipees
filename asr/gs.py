"""Electronic ground state properties."""
import typing
import numpy as np

from asr.core import command, option, DictStr, ASRResult
from asr.paneldata import GsResult, VacuumLevelResults, GapsResult


@command(module='asr.gs',
         creates=['gs.gpw'],
         requires=['structure.json'],
         resources='8:10h')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def calculate(calculator: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '200%',
        'txt': 'gs.txt',
        'charge': 0}) -> ASRResult:
    """Calculate ground state file.

    This recipe saves the ground state to a file gs.gpw based on the structure
    in 'structure.json'. This can then be processed by asr.gs@postprocessing
    for storing any derived quantities. See asr.gs@postprocessing for more
    information.
    """
    from ase.io import read
    from ase.calculators.calculator import PropertyNotImplementedError
    from asr.relax import set_initial_magnetic_moments
    atoms = read('structure.json')

    if not atoms.has('initial_magmoms'):
        set_initial_magnetic_moments(atoms)

    nd = np.sum(atoms.pbc)
    if nd == 2:
        assert not atoms.pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        calculator['poissonsolver'] = {'dipolelayer': 'xy'}

    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)

    atoms.calc = calc
    atoms.get_forces()
    try:
        atoms.get_stress()
    except PropertyNotImplementedError:
        pass
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw')

    return ASRResult()


def gaps(calc, soc=True) -> GapsResult:
    # ##TODO min kpt dens? XXX
    # inputs: gpw groundstate file, soc?, direct gap? XXX
    from functools import partial
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis

    if soc:
        ibzkpts = calc.get_bz_k_points()
    else:
        ibzkpts = calc.get_ibz_k_points()

    (evbm_ecbm_gap,
     skn_vbm, skn_cbm) = get_gap_info(soc=soc, direct=False,
                                      calc=calc)
    (evbm_ecbm_direct_gap,
     direct_skn_vbm, direct_skn_cbm) = get_gap_info(soc=soc, direct=True,
                                                    calc=calc)

    k_vbm, k_cbm = skn_vbm[1], skn_cbm[1]
    direct_k_vbm, direct_k_cbm = direct_skn_vbm[1], direct_skn_cbm[1]

    get_kc = partial(get_1bz_k, ibzkpts, calc)

    k_vbm_c = get_kc(k_vbm)
    k_cbm_c = get_kc(k_cbm)
    direct_k_vbm_c = get_kc(direct_k_vbm)
    direct_k_cbm_c = get_kc(direct_k_cbm)

    if soc:
        theta, phi = get_spin_axis()
        _, efermi = calc2eigs(calc, soc=True,
                              theta=theta, phi=phi)
    else:
        efermi = calc.get_fermi_level()

    return GapsResult.fromdata(
        gap=evbm_ecbm_gap[2],
        vbm=evbm_ecbm_gap[0],
        cbm=evbm_ecbm_gap[1],
        gap_dir=evbm_ecbm_direct_gap[2],
        vbm_dir=evbm_ecbm_direct_gap[0],
        cbm_dir=evbm_ecbm_direct_gap[1],
        k_vbm_c=k_vbm_c,
        k_cbm_c=k_cbm_c,
        k_vbm_dir_c=direct_k_vbm_c,
        k_cbm_dir_c=direct_k_cbm_c,
        skn1=skn_vbm,
        skn2=skn_cbm,
        skn1_dir=direct_skn_vbm,
        skn2_dir=direct_skn_cbm,
        efermi=efermi
    )


def get_1bz_k(ibzkpts, calc, k_index):
    from gpaw.kpt_descriptor import to1bz
    k_c = ibzkpts[k_index] if k_index is not None else None
    if k_c is not None:
        k_c = to1bz(k_c[None], calc.wfs.gd.cell_cv)[0]
    return k_c


def get_gap_info(soc, direct, calc):
    from ase.dft.bandgap import bandgap
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis
    # e1 is VBM, e2 is CBM
    if soc:
        theta, phi = get_spin_axis()
        e_km, efermi = calc2eigs(calc,
                                 soc=True, theta=theta, phi=phi)
        # km1 is VBM index tuple: (s, k, n), km2 is CBM index tuple: (s, k, n)
        gap, km1, km2 = bandgap(eigenvalues=e_km, efermi=efermi, direct=direct,
                                output=None)
        if km1[0] is not None:
            e1 = e_km[km1]
            e2 = e_km[km2]
        else:
            e1, e2 = None, None
        x = (e1, e2, gap), (0,) + tuple(km1), (0,) + tuple(km2)
    else:
        g, skn1, skn2 = bandgap(calc, direct=direct, output=None)
        if skn1[1] is not None:
            e1 = calc.get_eigenvalues(spin=skn1[0], kpt=skn1[1])[skn1[2]]
            e2 = calc.get_eigenvalues(spin=skn2[0], kpt=skn2[1])[skn2[2]]
        else:
            e1, e2 = None, None
        x = (e1, e2, g), skn1, skn2
    return x


def vacuumlevels(atoms, calc, n=8) -> VacuumLevelResults:
    """Get the vacuum level(s).

    Special case for 2D:

        Get the vacuumlevels on both sides of a material.
        Assumes the 2D material periodic directions are x and y.
        Assumes that the 2D material is centered in the z-direction of
        the unit cell.

    Parameters
    ----------
    atoms: Atoms
       The Atoms object.
    calc: GPAW-calculator
        The GPAW object.  Provides the electrostatic potential.
    n: int
        Number of gridpoints away from the edge to evaluate the vacuum levels.
    """
    import numpy as np

    z_z = None
    v_z = None
    devac = None
    dipz = None
    evac1 = None
    evac2 = None
    evacmean = None

    ves = calc.get_electrostatic_potential()

    nperiodic = atoms.pbc.sum()
    if nperiodic < 2:
        evacmean = 0.0
        for v, periodic in zip([ves[0], ves[:, 0], ves[:, :, 0]],
                               atoms.pbc):
            if not periodic:
                evacmean += v.mean() / (3 - nperiodic)
    elif nperiodic == 2:
        # Record electrostatic potential as a function of z
        assert not atoms.pbc[2]
        v_z = ves.mean(0).mean(0)
        z_z = np.linspace(0, atoms.cell[2, 2], len(v_z), endpoint=False)
        dipz = atoms.get_dipole_moment()[2]
        devac = evacdiff(atoms)
        evac1 = v_z[n]
        evac2 = v_z[-n]
        evacmean = (v_z[n] + v_z[-n]) / 2

    return VacuumLevelResults.fromdata(
        z_z=z_z,
        v_z=v_z,
        dipz=dipz,
        evacdiff=devac,
        evac1=evac1,
        evac2=evac2,
        evacmean=evacmean,
        efermi_nosoc=calc.get_fermi_level())


def evacdiff(atoms):
    """Derive vacuum energy level difference from the dipole moment.

    Calculate vacuum energy level difference from the dipole moment of
    a slab assumed to be in the xy plane

    Returns
    -------
    out: float
        vacuum level difference in eV
    """
    import numpy as np
    from ase.units import Bohr, Hartree

    A = np.linalg.det(atoms.cell[:2, :2] / Bohr)
    dipz = atoms.get_dipole_moment()[2] / Bohr
    evacsplit = 4 * np.pi * dipz / A * Hartree

    return evacsplit


@command(module='asr.gs',
         requires=['gs.gpw', 'structure.json',
                   'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.gs@calculate', 'asr.magnetic_anisotropy',
                       'asr.structureinfo'],
         returns=GsResult)
def main() -> GsResult:
    """Extract derived quantities from groundstate in gs.gpw."""
    from ase.io import read
    from asr.calculators import get_calculator

    # Just some quality control before we start
    atoms = read('structure.json')
    calc = get_calculator()('gs.gpw')
    pbc = atoms.pbc
    ndim = np.sum(pbc)

    if ndim == 2:
        assert not pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        # For 2D materials we check that the calculater used a dipole
        # correction if the material has an out-of-plane dipole

        # Small hack
        atoms = calc.atoms
        atoms.calc = calc
        evacdiffmin = 10e-3
        if evacdiff(calc.atoms) > evacdiffmin:
            assert calc.todict().get('poissonsolver', {}) == \
                {'dipolelayer': 'xy'}, \
                ('The ground state has a finite dipole moment along aperiodic '
                 'axis but calculation was without dipole correction.')

    # Now that some checks are done, we can extract information
    forces = calc.get_property('forces', allow_calculation=False)
    stresses = calc.get_property('stress', allow_calculation=False)
    etot = calc.get_potential_energy()

    gaps_nosoc = gaps(calc, soc=False)
    gaps_soc = gaps(calc, soc=True)
    vac = vacuumlevels(atoms, calc)
    workfunction = vac.evacmean - gaps_soc.efermi if vac.evacmean else None
    return GsResult.fromdata(
        forces=forces,
        stresses=stresses,
        etot=etot,
        gaps_nosoc=gaps_nosoc,
        gap_dir_nosoc=gaps_nosoc.gap_dir,
        gap_nosoc=gaps_nosoc.gap,
        gap=gaps_soc.gap,
        vbm=gaps_soc.vbm,
        cbm=gaps_soc.cbm,
        gap_dir=gaps_soc.gap_dir,
        vbm_dir=gaps_soc.vbm_dir,
        cbm_dir=gaps_soc.cbm_dir,
        k_vbm_c=gaps_soc.k_vbm_c,
        k_cbm_c=gaps_soc.k_cbm_c,
        k_vbm_dir_c=gaps_soc.k_vbm_dir_c,
        k_cbm_dir_c=gaps_soc.k_cbm_dir_c,
        skn1=gaps_soc.skn1,
        skn2=gaps_soc.skn2,
        skn1_dir=gaps_soc.skn1_dir,
        skn2_dir=gaps_soc.skn2_dir,
        efermi=gaps_soc.efermi,
        vacuumlevels=vac,
        dipz=vac.dipz,
        evac=vac.evacmean,
        evacdiff=vac.evacdiff,
        workfunction=workfunction)


if __name__ == '__main__':
    main.cli()
