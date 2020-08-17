from asr.core import command, argument, AtomsFromFile, ASRCalculator, CalcFromStr, \
    ASRResults
from ase import Atoms
from asr.magnetic_anisotropy import main as magnetic_anisotropy
from asr.structureinfo import main as structureinfo
import numpy as np
from typing import Tuple


def resources(parameters):
    """Get resources of ground state calculation."""
    return dict(cores=8, time='10h')


class ASRGSCalculateResults(ASRResults):
    """Results of asr.gs@calculate."""

    gs_filename: str
    side_effects = ['gs_filename']
    desriptions = {'gs_filename': 'Ground state file name.'}


@command(namespace='asr.gs',
         resources=resources,
         version=0)
@argument("atoms",
          help='Atomic structure.',
          type=Atoms,
          cli_argtype='option',
          cli_typecast=AtomsFromFile(),
          cli_default='structure.json')
@argument('calculator',
          help='Calculator params.',
          type=ASRCalculator,
          has_package_dependencies=True,
          cli_argtype='option',
          cli_typecast=CalcFromStr())
@argument('filename',
          help='Filename of groundstate file.',
          type=str,
          side_effect=True,
          cli_argtype='option',
          cli_default='gs.gpw')
def calculate(
        atoms: Atoms,
        calculator: ASRCalculator = ASRCalculator(
            name='gpaw',
            mode={'name': 'pw', 'ecut': 800},
            xc='PBE',
            basis='dzp',
            kpts={'density': 12.0, 'gamma': True},
            occupations={'name': 'fermi-dirac',
                         'width': 0.05},
            convergence={'bands': 'CBM+3.0'},
            nbands='200%',
            txt='gs.txt',
            charge=0),
        filename: str = 'gs.gpw'
) -> ASRGSCalculateResults:
    """Calculate ground state file.

    This recipe saves the ground state to a file gs.gpw based on the structure
    in 'structure.json'. This can then be processed by asr.gs@postprocessing
    for storing any derived quantities. See asr.gs@postprocessing for more
    information.
    """
    from ase.calculators.calculator import PropertyNotImplementedError
    from asr.relax import set_initial_magnetic_moments

    if not atoms.has('initial_magmoms'):
        set_initial_magnetic_moments(atoms)

    atoms.calc = calculator
    atoms.get_forces()
    try:
        atoms.get_stress()
    except PropertyNotImplementedError:
        pass
    atoms.get_potential_energy()
    atoms.calc.write(filename)

    return ASRGSCalculateResults(gs_filename=filename)


def webpanel(row, key_descriptions):
    from asr.database.browser import table, fig

    t = table(row, 'Property',
              ['gap', 'gap_dir',
               'dipz', 'evacdiff', 'workfunction', 'dos_at_ef_soc'],
              key_descriptions)

    gap = row.get('gap')

    if gap > 0:
        if row.get('evac'):
            t['rows'].extend(
                [['Valence band maximum wrt. vacuum level',
                  f'{row.vbm - row.evac:.2f} eV'],
                 ['Conduction band minimum wrt. vacuum level',
                  f'{row.cbm - row.evac:.2f} eV']])
        else:
            t['rows'].extend(
                [['Valence band maximum wrt. Fermi level',
                  f'{row.vbm - row.efermi:.2f} eV'],
                 ['Conduction band minimum wrt. Fermi level',
                  f'{row.cbm - row.efermi:.2f} eV']])
    panel = {'title': 'Basic electronic properties (PBE)',
             'columns': [[t], [fig('bz-with-gaps.png')]],
             'sort': 10}

    row = ['Band gap (PBE)', f'{row.gap:0.2f} eV']
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Electronic properties', ''],
                             'rows': [row]}]],
               'plot_descriptions': [{'function': bz_soc,
                                      'filenames': ['bz-with-gaps.png']}],
               'sort': 10}

    return [panel, summary]


def bz_soc(row, fname):
    from ase.geometry.cell import Cell
    from matplotlib import pyplot as plt
    import numpy as np
    cell = Cell(row.cell)
    lat = cell.get_bravais_lattice(pbc=row.pbc)
    plt.figure(figsize=(4, 4))
    lat.plot_bz(vectors=False, pointstyle={'c': 'k', 'marker': '.'})
    gsresults = row.data.get('results-asr.gs.json')
    cbm_c = gsresults['k_cbm_c']
    vbm_c = gsresults['k_vbm_c']
    op_scc = row.data[
        'results-asr.structureinfo.json']['spglib_dataset']['rotations']
    if cbm_c is not None:
        ax = plt.gca()
        icell_cv = np.linalg.inv(row.cell).T
        vbm_style = {'marker': 'o', 'facecolor': 'w',
                     'edgecolors': 'C0', 's': 100, 'lw': 2,
                     'zorder': 4}
        cbm_style = {'c': 'C1', 'marker': 'o', 's': 40, 'zorder': 5}
        cbm_sc = np.dot(op_scc.transpose(0, 2, 1), cbm_c)
        vbm_sc = np.dot(op_scc.transpose(0, 2, 1), vbm_c)
        cbm_sv = np.dot(cbm_sc, icell_cv)
        vbm_sv = np.dot(vbm_sc, icell_cv)
        ax.scatter([vbm_sv[:, 0]], [vbm_sv[:, 1]], **vbm_style, label='VBM')
        ax.scatter([cbm_sv[:, 0]], [cbm_sv[:, 1]], **cbm_style, label='CBM')
        xlim = np.array(ax.get_xlim()) * 1.4
        ylim = np.array(ax.get_ylim()) * 1.4
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.legend(loc='upper center', ncol=3)

    plt.tight_layout()
    plt.savefig(fname)


class GapResults(ASRResults):
    """Band gap results."""

    gap: float
    gap_description: str = 'The band gap [eV].'
    vbm: float
    vbm_description: str = 'The valence band maximum [eV].'
    cbm: float
    cbm_description: str = 'The conduction band minimum [eV].'
    gap_dir: float
    gap_dir_description: str = 'The direct band gap [eV].'
    vbm_dir: float
    vbm_dir_description: str = 'The direct valence band maximum [eV].'
    cbm_dir: float
    cbm_dir_description: str = 'The direct conduction band minimum [eV].'
    k_vbm_c: Tuple[float, float, float]
    k_vbm_c_description: str = (
        'The k-point coordinate of the valence band maximum '
        'in units of the reciprocal lattice vectors.'
    )
    k_cbm_c: Tuple[float, float, float]
    k_cbm_c_description: str = (
        'The k-point coordinate of the conduction band minimum '
        'in units of the reciprocal lattice vectors.'
    )
    k_vbm_dir_c: Tuple[float, float, float]
    k_vbm_c_description: str = (
        'The k-point coordinate of the direct valence band maximum '
        'in units of the reciprocal lattice vectors.'
    )
    k_cbm_dir_c: Tuple[float, float, float]
    k_cbm_c_description: str = (
        'The k-point coordinate of the direct conduction band minimum '
        'in units of the reciprocal lattice vectors.'
    )
    skn1: Tuple[int, int, int]
    skn1_description: str = (
        '(spin, k-point, band) indices of the valence band maximum.'
    )
    skn2: Tuple[int, int, int]
    skn2_description: str = (
        '(spin, k-point, band) indices of the conduction band minimum.'
    )
    skn1_dir: Tuple[int, int, int]
    skn1_description: str = (
        '(spin, k-point, band) indices of the direct valence band maximum.'
    )
    skn2_dir: Tuple[int, int, int]
    skn2_dir_description: str = \
        '(spin, k-point, band) indices of the direct conduction band minimum.'

    efermi: float
    efermi_description: str = 'The Fermi level [eV].'


class ASRGSResults(ASRResults):
    """Results of the ground state recipe."""

    forces: np.ndarray
    forces_description = 'Forces on atoms [eV/Angstrom]'
    stresses: np.ndarray
    stresses_description: str = 'Stress on unit cell [eV/Angstrom^dim]'
    etot: float
    etot_description: str = 'Total energy (Tot. En.) [eV]'
    evac: float
    evac_description: str = 'Vacuum level (Vacuum level) [eV]'
    evacdiff: float
    evacdiff_description: str = \
        'Vacuum level shift (Vacuum level shift) [eV]'
    dipz_description: str = 'Out-of-plane dipole [e * Ang]'
    efermi_description: str = 'Fermi level (Fermi level) [eV]'
    gap_description: str = 'Band gap (Band gap) [eV]'
    vbm_description: str = 'Valence band maximum (Val. band max.) [eV]'
    cbm_description: str = 'Conduction band minimum (Cond. band max.) [eV]'
    gap_dir_description: str = 'Direct band gap (Dir. band gap) [eV]'
    vbm_dir_description: str = \
        'Direct valence band maximum (Dir. val. band max.) [eV]'
    cbm_dir_description: str = \
        'Direct conduction band minimum (Dir. cond. band max.) [eV]'
    gap_dir_nosoc: str = \
        'Direct gap without SOC (Dir. gap wo. soc.) [eV]'
    gaps_nosoc: GapResults
    gaps_nosoc_description: str = \
        'Band gap results without spin-out correction.'


@command(namespace='asr.gs',
         dependencies=[calculate,
                       magnetic_anisotropy,
                       structureinfo],
         returns=ASRGSResults)
def main() -> ASRGSResults:
    """Extract derived quantities from groundstate in gs.gpw."""
    import numpy as np
    from asr.calculators import get_calculator
    from gpaw.mpi import serial_comm

    # Just some quality control before we start
    calculate_results = calculate()
    gsfile = calculate_results.parameters.filename
    calc = get_calculator()(gsfile,
                            txt=None,
                            communicator=serial_comm)
    atoms = calc.atoms
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
    gap_dir_nosoc = gaps_nosoc['gap_dir']
    gap_nosoc = gaps_nosoc['gap']

    vac = vacuumlevels(atoms, calc)
    results['vacuumlevels'] = vac
    dipz = vac['dipz']
    evac = vac['evacmean']
    evacdiff = vac['evacdiff']
    workfunction = results['evac'] - results['efermi']

    return ASRGSResults(forces=forces,
                        stresses=stresses,
                        etot=etot,
                        evac=evac,
                        evacdiff=evacdiff,
                        dipz=dipz,
                        efermi=gaps_soc.efermi,
                        gap=gaps_soc.gap,
                        vbm=gaps_soc.vbm,
                        cbm=gaps_soc.cbm,
                        gap_dir=gaps_soc.gap_dir,
                        vbm_dir=gaps_soc.vbm_dir,
                        cbm_dir=gaps_soc.cbm_dir,
                        gap_dir_nosoc=gap_dir_nosoc)



def gaps(calc, soc=True) -> GapResults:
    # ##TODO min kpt dens? XXX
    # inputs: gpw groundstate file, soc?, direct gap? XXX
    from functools import partial
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis

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
        _, efermi = calc2eigs(calc, ranks=[0], soc=True,
                              theta=theta, phi=phi)
    else:
        efermi = calc.get_fermi_level()

    subresults = {'gap': evbm_ecbm_gap[2],
                  'vbm': evbm_ecbm_gap[0],
                  'cbm': evbm_ecbm_gap[1],
                  'gap_dir': evbm_ecbm_direct_gap[2],
                  'vbm_dir': evbm_ecbm_direct_gap[0],
                  'cbm_dir': evbm_ecbm_direct_gap[1],
                  'k_vbm_c': k_vbm_c,
                  'k_cbm_c': k_cbm_c,
                  'k_vbm_dir_c': direct_k_vbm_c,
                  'k_cbm_dir_c': direct_k_cbm_c,
                  'skn1': skn_vbm,
                  'skn2': skn_cbm,
                  'skn1_dir': direct_skn_vbm,
                  'skn2_dir': direct_skn_cbm,
                  'efermi': efermi}

    return subresults


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
        e_km, efermi = calc2eigs(calc, ranks=[0],
                                 soc=True, theta=theta, phi=phi)
        # km1 is VBM index tuple: (s, k, n), km2 is CBM index tuple: (s, k, n)
        gap, km1, km2 = bandgap(eigenvalues=e_km, efermi=efermi, direct=direct,
                                kpts=calc.get_ibz_k_points(), output=None)
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


def vacuumlevels(atoms, calc, n=8):
    """Get the vacuumlevels on both sides of a 2D material.

    Get the vacuumlevels on both sides of a 2D material. Will
    do a dipole corrected dft calculation, if needed (Janus structures).
    Assumes the 2D material periodic directions are x and y.
    Assumes that the 2D material is centered in the z-direction of
    the unit cell.

    Dipole corrected dft calculation -> dipcorrgs.gpw

    Parameters
    ----------
    gpw: str
       name of gpw file to base the dipole corrected calc on
    evacdiffmin: float
        thresshold in eV for doing a dipole moment corrected
        dft calculations if the predicted evac difference is less
        than this value don't do it
    n: int
        number of gridpoints away from the edge to evaluate the vac levels
    """
    import numpy as np

    # Record electrostatic potential as a function of z
    v_z = calc.get_electrostatic_potential().mean(0).mean(0)
    z_z = np.linspace(0, atoms.cell[2, 2], len(v_z), endpoint=False)

    # Store data
    subresults = {'z_z': z_z, 'v_z': v_z,
                  'evacdiff': evacdiff(atoms),
                  'dipz': atoms.get_dipole_moment()[2],
                  # Extract vaccuum energy on both sides of the slab
                  'evac1': v_z[n],
                  'evac2': v_z[-n],
                  'evacmean': (v_z[n] + v_z[-n]) / 2,
                  # Ef might have changed in dipole corrected gs
                  'efermi_nosoc': calc.get_fermi_level()}

    return subresults


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


if __name__ == '__main__':
    main.cli()
