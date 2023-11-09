import numpy as np
from asr.core import command, option
from asr.result.resultdata import ShgResult, CentroSymmetric


def get_chi_symmetry(atoms, sym_th=None):

    # Get the symmetry of the structure and operations
    import spglib
    from asr.utils.symmetry import c2db_symmetry_eps, c2db_symmetry_angle

    if sym_th is None:
        sym_th = c2db_symmetry_eps

    sg = spglib.get_symmetry((atoms.cell,
                              atoms.get_scaled_positions(),
                              atoms.get_atomic_numbers()),
                             symprec=sym_th,
                             angle_tolerance=c2db_symmetry_angle)
    op_scc = sg['rotations']

    # Make a random symmetrized matrix
    chi_vvv = 1 + np.random.rand(3, 3, 3)
    for v1 in range(3):
        chi_vvv[v1] = (chi_vvv[v1] + chi_vvv[v1].T) / 2.0

    # Introduce the symmetries to the matrix
    cell_cv = atoms.cell
    op_svv = [np.linalg.inv(cell_cv).dot(op_cc.T).dot(cell_cv) for
              op_cc in op_scc]
    nop = len(op_svv)
    sym_chi_vvv = np.zeros_like(chi_vvv)
    for op_vv in op_svv:
        sym_chi_vvv += np.einsum('il,jm,kn,lmn->ijk',
                                 op_vv, op_vv, op_vv, chi_vvv)
    sym_chi_vvv /= nop

    # Make the symmetry tensor dictionary
    sym_chi = {'zero': ''}
    ind_list = list(range(27))
    ind_list[1], ind_list[13] = ind_list[13], ind_list[1]
    nz_pols = []
    for ii, ind in enumerate(ind_list):
        v1, v2, v3 = int(ind / 9), int((ind % 9) / 3), (ind % 9) % 3
        pol = 'xyz'[v1] + 'xyz'[v2] + 'xyz'[v3]
        if not np.isclose(sym_chi_vvv[v1, v2, v3], 0.0):
            nz_pols.append(pol)
            sym_chi[pol] = pol
            for indc in ind_list[ii + 1:]:
                v1c, v2c = int(indc / 9), int((indc % 9) / 3)
                v3c = (indc % 9) % 3
                polc = 'xyz'[v1c] + 'xyz'[v2c] + 'xyz'[v3c]
                if np.isclose(sym_chi_vvv[v1, v2, v3],
                              sym_chi_vvv[v1c, v2c, v3c]):
                    sym_chi_vvv[v1c, v2c, v3c] = 0.0
                    sym_chi[pol] += '=' + polc
                    nz_pols.append(polc)
                elif np.isclose(sym_chi_vvv[v1, v2, v3],
                                -sym_chi_vvv[v1c, v2c, v3c]):
                    sym_chi_vvv[v1c, v2c, v3c] = 0.0
                    sym_chi[pol] += '=-' + polc
                    nz_pols.append(polc)
        else:
            if pol not in nz_pols:
                sym_chi['zero'] += '=' + pol
    sym_chi['zero'] = sym_chi['zero'][1:]

    # Check the number of elements
    if sym_chi['zero'] != '':
        nr_el = len(sym_chi['zero'].split('='))
    else:
        nr_el = 0
    if nr_el + len(list(set(nz_pols))) != 27:
        print('Something is wrong with symmetry!')

    return sym_chi


def get_kpts(kptdensity, nd, cell):
    kpts = {}
    if nd == 3 or nd == 1:
        kpts = {'density': kptdensity, 'gamma': False, 'even': True}
    elif nd == 2:
        vx, vy, _ = cell
        lx, ly = np.sqrt(np.sum(vx**2)), np.sqrt(np.sum(vy**2))
        kx_gs = kptdensity * 2.0 * np.pi / lx
        ky_gs = kptdensity * 2.0 * np.pi / ly
        kx_gs = int(kx_gs) - int(kx_gs) % 2
        ky_gs = int(ky_gs) - int(ky_gs) % 2
        kpts = {'size': (kx_gs, ky_gs, 1), 'gamma': True}

    return kpts


@command('asr.shg',
         dependencies=['asr.gs@calculate'],
         requires=['structure.json', 'gs.gpw'],
         returns=ShgResult)
@option('--gs', help='Ground state on which response is based',
        type=str)
@option('--kptdensity', help='K-point density [1/Ang]', type=float)
@option('--gauge', help='Selected gauge (length "lg" or velocity "vg")',
        type=str)
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor')
@option('--eta', help='Broadening [eV]', type=float)
@option('--maxomega', help='Max pump frequency [eV]', type=float)
@option('--nromega', help='Number of pump frequencies', type=int)
@option('--removefiles', help='Remove created files', type=bool)
def main(gs: str = 'gs.gpw', kptdensity: float = 25.0, gauge: str = 'lg',
         bandfactor: int = 4, eta: float = 0.05,
         maxomega: float = 10.0, nromega: int = 1000,
         removefiles: bool = False) -> ShgResult:
    """Calculate the SHG spectrum, only independent tensor elements.

    The recipe computes the SHG spectrum. The tensor in general have 18 independent
    tensor elements (since it is symmetric). However, the point group symmety reduces
    the number of independent tensor elements.
    The SHG spectrum is calculated using perturbation theory, where the perturbation
    can be written in either the length gauge (r.E) or velocity gauge (p.A).

    Parameters
    ----------
    gs : str
        The ground state filename.
    kptdensity : float
        K-point density.
    gauge : str
        Selected gauge (length "lg" or velocity "vg").
    bandfactor : int
        Number of unoccupied bands: (#occ. bands) * bandfactor.
    eta : float
        Broadening used for finding the spectrum.
    maxomega : float
        Max pump frequency.
    nromega : int
        Number of pump frequencies.
    removefiles : bool
        Remove intermediate files that are created.
    """
    from ase.io import read
    from gpaw import GPAW
    from gpaw.mpi import world
    from pathlib import Path
    from gpaw.nlopt.matrixel import make_nlodata
    from gpaw.nlopt.shg import get_shg

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
        # fnames = ['es.gpw', 'mml.npz']
        fnames = []
        mml_name = 'mml.npz'
        if not Path(mml_name).is_file():
            if not Path('es.gpw').is_file():
                calc_old = GPAW(gs, txt=None)
                nval = calc_old.wfs.nvalence

                calc = GPAW(
                    gs,
                    txt='es.txt',
                    symmetry={'point_group': False, 'time_reversal': True},
                    fixdensity=True,
                    nbands=(bandfactor + 1) * nval,
                    convergence={'bands': bandfactor * nval},
                    occupations={'name': 'fermi-dirac', 'width': 1e-4},
                    kpts=kpts)
                calc.get_potential_energy()
                calc.write('es.gpw', mode='all')
                fnames.append('es.gpw')

            # Calculate momentum matrix:
            make_nlodata(gs_name='es.gpw', out_name=mml_name)
            fnames.append(mml_name)

        # Do the calculation
        chi_dict = {}
        for pol in sorted(sym_chi.keys()):
            if pol == 'zero':
                continue
            # Do the SHG calculation
            shg_name = 'shg_{}.npy'.format(pol)
            if not Path(shg_name).is_file():
                shg = get_shg(
                    freqs=w_ls, eta=eta, pol=pol, gauge=gauge,
                    out_name=shg_name, mml_name=mml_name)
            else:
                shg = np.load(shg_name)

            # Make the output data
            fnames.append(shg_name)
            if nd == 3:
                chi_dict[pol] = shg[1] * 1e9
            else:
                # Make it a surface chi instead of bulk chi
                cellsize = atoms.cell.cellpar()
                chi_dict[pol] = shg[1] * cellsize[2] * 1e8

        # Make the output data
        results = {
            'chi': chi_dict,
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
