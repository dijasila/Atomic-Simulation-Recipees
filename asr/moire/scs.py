from typing import Union
import numpy as np
from gpaw import GPAW
from gpaw.lcao.scissors import Scissors
from gpaw.spinorbit import soc_eigenstates
from ase.io import read
from ase.io.jsonio import read_json, write_json
from ase.dft.bandgap import bandgap
from asr.core import command, option, DictStr, ASRResult, prepare_result


def calculate_evac(calc):
    """Obtain vacuum level from a GPAW calculator"""
    return np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]


def get_kpts_size(atoms, density):
    """Try to get a reasonable monkhorst size which hits high symmetry points."""
    from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
    size, _ = k2so(atoms=atoms, density=density)
    size[2] = 1
    for i in range(2):
        if size[i] % 6 != 0:
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts


def is_almost_hexagonal(atoms):
    """ Returns True if the angle between the a and b cell vectors is close to 60
        (1 degree tolerance) and their norm is equal (1% tolerance)
    """
    from asr.findmoire import angle_between
    cell_0 = atoms.cell[0]
    cell_1 = atoms.cell[1]
    angle = angle_between(cell_0, cell_1)
    if np.isclose(angle, np.pi / 3, atol=0, rtol=1.7e-2) and \
       np.isclose(np.linalg.norm(cell_0), np.linalg.norm(cell_1), atol=0, rtol=1e-2):
        return True
    return False


def update_with_soc(dct, calc):
    """ Calculates spin-orbit coupling, 
        dumps eigenvalues, gaps and band edges to dictionary
    """
    calc_soc = soc_eigenstates(calc)
    ev_soc = calc_soc.eigenvalues()
    ef_soc = calc_soc.fermi_level
    hl_soc, vbm_k_soc, cbm_k_soc = bandgap(eigenvalues=ev_soc, efermi=ef_soc, direct=False)
    dir_soc, _, _ = bandgap(eigenvalues=ev_soc, efermi=ef_soc, direct=True)

    dct.update({
        'efermi_soc': ef_soc,
        'hl_gap_soc': hl_soc,
        'dir_gap_soc': dir_soc,
        'vbm_soc': ev_soc[vbm_k_soc[0], vbm_k_soc[1]],
        'cbm_soc': ev_soc[cbm_k_soc[0], cbm_k_soc[1]],
        'vbm_k_soc': vbm_k_soc,
        'cbm_k_soc': cbm_k_soc,
        'eigenvalues_soc': ev_soc,
        'spin_projections': calc_soc.spin_projections()
    })
    return dct


def dump_to_json(filename, calc, soc, bs):
    """ Calculates Spin-orbit coupling if requested, 
        dumps electronic structure information to lightweight JSON file.
    """
    ibzkpts = calc.get_ibz_k_points()
    bzmap = calc.get_bz_to_ibz_map()
    kpts = np.asarray([ibzkpts[i] for i in bzmap])
    hl_nosoc, vbm_k, cbm_k = bandgap(calc, direct=False)
    dir_nosoc, _, _ = bandgap(calc, direct=True)
    ev_nosoc = np.asarray([calc.get_eigenvalues(kpt=i) for i in bzmap])

    dct = {
        'efermi_nosoc': calc.get_fermi_level(),
        'evac': calculate_evac(calc),
        'hl_gap_nosoc': hl_nosoc,
        'dir_gap_nosoc': dir_nosoc,
        'vbm_nosoc': calc.get_homo_lumo()[0],
        'cbm_nosoc': calc.get_homo_lumo()[1],
        'vbm_k_nosoc': vbm_k[1:],
        'cbm_k_nosoc': cbm_k[1:],
        'eigenvalues_nosoc': ev_nosoc,
        'kpts': kpts
    }

    if soc:
        dct = update_with_soc(dct, calc)

    if bs:
        bsdct = calc.band_structure().__dict__.copy()
        dct.update({'path': bsdct.pop('_path')})

    write_json(filename, dct)
    return 0


def get_scissors_operator(atoms, shifts):
    """ Returns the scissors operator already set up with shifts """
    shft = read_json(shifts)
    tags = atoms.get_tags()
    n_upper = len(tags[tags == 1])
    n_lower = len(tags[tags == 0])
    return Scissors([(shft['shift_v1'], shft['shift_c1'], n_upper),
                     (shft['shift_v2'], shft['shift_c2'], n_lower)])


def calculate_gs(atoms, kpts, calculator, scs):
    """ Calculates the SCS ground state """
    kpts = get_kpts_size(atoms, kpts)
    calculator.update({'kpts': kpts})
    if scs:
        calculator.update({'eigensolver': scs})
        filename = 'gs_scs.gpw'
    else:
        filename = 'gs_pbe.gpw'
    calc = GPAW(**calculator)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.calc.write(filename, mode='')
    return calc


def calculate_bs(gpw, kptpath, npoints, eps, scs):
    "Calculate electronic band structure with the self-consistent scissors corrections"
    calc = GPAW(gpw, txt=None)
    if not kptpath:
        path = calc.atoms.cell.bandpath(npoints=npoints, pbc=calc.atoms.pbc, eps=eps)
        if path.path == 'GXA1YG' and is_almost_hexagonal(calc.atoms):
            path = calc.atoms.cell.bandpath(path='GYA1G', npoints=npoints,
                                       pbc=calc.atoms.pbc, eps=eps)
    else:
        path = calc.atoms.cell.bandpath(path=kptpath, npoints=npoints,
                                   pbc=calc.atoms.pbc, eps=eps)
    parms = {
        'basis': 'dzp',
        'txt': 'bs_scs.txt',
        'fixdensity': True,
        'symmetry': 'off',
        'kpts': {
            'path': path.path,
            'npoints': npoints,
            'eps': eps
        }
    }
    if scs:
        parms.update({'eigensolver': scs})
        filename = 'gs_scs.gpw'
    else:
        filename = 'gs_pbe.gpw'
    calc.set(**parms)
    calc.get_potential_energy()
    calc.write(filename, mode='')
    return calc


@command('asr.scs')
@option('--structure')
@option('--shifts')
@option('--kptpath', help='Custom kpoint path.')
@option('--npoints')
@option('--kpts', help="In-plane kpoint density")
@option('--gs', is_flag=True, help='Request only ground state calculation')
@option('--bs', is_flag=True, help='Request only band structure calculation')
@option('--soc', is_flag=True, help='Calculate spin-orbit coupling')
@option('--no-scs', is_flag=True, help='Perform a plain DFT-PBE calculation in the LCAO basis without SCS shifts.')
@option('--eps', help='Tolerance over symmetry determination')
@option('--calculator', help="Dictionary containing calculator parameters")
@option('--gpw', help="Path to existing .gpw file")
def main(structure: str = 'structure.json',
         shifts: str = 'shifts.json',
         kptpath: Union[str, None] = None,
         npoints: int = 200,
         kpts: int = 12,
         gs: bool = False,
         bs: bool = False,
         soc: bool = False,
         no_scs: bool = False,
         eps: float = 2e-4,
         gpw: str = 'gs_scs.gpw',
         calculator: dict = {
             'mode': {'name': 'lcao'},
             'xc': 'PBE',
             'basis': 'dzp',
             'kpts': {'density': 12.0, 'gamma': True},
             'occupations': {'name': 'fermi-dirac',
                             'width': 0.05},
             'nbands': 'nao',
             'txt': 'gs_scs.txt',
             'maxiter': 333,
             'charge': 0}) -> ASRResult:

    atoms = read(structure)
    if no_scs:
        SCS = None
    else:
        SCS = get_scissors_operator(atoms, shifts)

    if gs:
        SCS = get_scissors_operator(atoms, shifts)
        calc = calculate_gs(atoms, kpts, calculator, SCS)
        if no_scs:
            filename = 'gs_pbe.json'
        else:
            filename = 'gs_scs.json'
        dump_to_json(filename, calc, soc, bs)

    if bs:
        SCS = get_scissors_operator(atoms, shifts)
        calc = calculate_bs(gpw, kptpath, npoints, eps, SCS)
        if no_scs:
            filename = 'bs_pbe.json'
        else:
            filename = 'bs_scs.json'
        dump_to_json(filename, calc, soc, bs)


if __name__ == "__main__":
    main.cli()
