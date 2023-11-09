import numpy as np
from asr.core import command, option
from asr.shg import CentroSymmetric, get_chi_symmetry, get_kpts
from asr.result.resultdata import ShiftResult


@command('asr.shift',
         dependencies=['asr.gs@calculate'],
         requires=['structure.json', 'gs.gpw'],
         returns=ShiftResult)
@option('--gs', help='Ground state on which response is based',
        type=str)
@option('--kptdensity', help='K-point density', type=float)
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
@option('--eta', help='Broadening [eV]', type=float)
@option('--maxomega', help='Max frequency [eV]', type=float)
@option('--nromega', help='Number of frequencies', type=int)
@option('--energytol', help='Energy tolerance [eV]', type=float)
@option('--removefiles', help='Remove created files', type=bool)
def main(gs: str = 'gs.gpw', kptdensity: float = 25.0,
         bandfactor: int = 4, eta: float = 0.05, energytol: float = 1e-2,
         maxomega: float = 10.0, nromega: int = 1000,
         removefiles: bool = True) -> ShiftResult:
    """Calculate the shift current spectrum, only independent tensor elements.

    The recipe computes the shift current. The tensor in general have 18 independent
    tensor elements (since it is symmetric). However, the point group symmetry reduces
    the number of independent tensor elements.
    The shift spectrum is calculated using perturbation theory.

    Parameters
    ----------
    gs : str
        The ground state filename.
    kptdensity : float
        K-point density.
    bandfactor : int
        Number of unoccupied bands: (#occ. bands) * bandfactor.
    eta : float
        Broadening used for finding the spectrum.
    energytol : float
        Energy tolerance to remove degeneracies.
    maxomega : float
        Max frequency.
    nromega : int
        Number of frequencies.
    removefiles : bool
        Remove intermediate files that are created.
    """
    from ase.io import read
    from gpaw import GPAW
    from gpaw.mpi import world
    from pathlib import Path
    from gpaw.nlopt.matrixel import make_nlodata
    from gpaw.nlopt.shift import get_shift

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    nd = np.sum(pbc)
    kpts = get_kpts(kptdensity, nd, atoms.cell)
    sym_chi = get_chi_symmetry(atoms)

    # If the structure has inversion symmetry do nothing
    if len(sym_chi) == 1:
        raise CentroSymmetric

    w_ls = np.linspace(0, maxomega, nromega)
    try:
        fnames = []
        mml_name = 'mml.npz'
        if not Path(mml_name).is_file():
            if not Path('gs_shift.gpw').is_file():
                calc_old = GPAW(gs, txt=None)
                nval = calc_old.wfs.nvalence

                calc = GPAW(
                    gs,
                    txt='gs_shift.txt',
                    symmetry={'point_group': False, 'time_reversal': True},
                    fixdensity=True,
                    nbands=(bandfactor + 1) * nval,
                    convergence={'bands': bandfactor * nval},
                    occupations={'name': 'fermi-dirac', 'width': 1e-4},
                    kpts=kpts)
                calc.get_potential_energy()
                calc.write('gs_shift.gpw', mode='all')
                fnames.append('gs_shift.gpw')

            # Calculate momentum matrix:
            make_nlodata(gs_name='gs_shift.gpw', out_name=mml_name)
            fnames.append(mml_name)

        # Do the calculation
        sigma_dict = {}
        for pol in sorted(sym_chi.keys()):
            if pol == 'zero':
                continue
            # Do the shift current calculation
            shift_name = 'shift_{}.npy'.format(pol)
            if not Path(shift_name).is_file():
                shift = get_shift(
                    freqs=w_ls, eta=eta, pol=pol, Etol=energytol,
                    out_name=shift_name, mml_name=mml_name)
            else:
                shift = np.load(shift_name)

            # Make the output data
            # fnames.append(shift_name)
            if nd == 3:
                sigma_dict[pol] = shift[1] * 1e6
            else:
                # Make it a surface chi instead of bulk chi
                cellsize = atoms.cell.cellpar()
                sigma_dict[pol] = shift[1] * cellsize[2] * 1e5

        # Make the output data
        results = {
            'sigma': sigma_dict,
            'symm': sym_chi,
            'freqs': w_ls, }

    finally:
        world.barrier()
        if world.rank == 0 and removefiles:
            for filename in fnames:
                es_file = Path(filename)
                if es_file.is_file():
                    es_file.unlink()

    return results


if __name__ == '__main__':
    main.cli()
