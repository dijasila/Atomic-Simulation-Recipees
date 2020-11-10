import typing
from asr.core import command, option, ASRResult, prepare_result
import numpy as np


def get_chi_symmtery(atoms, sym_th=1e-3):
    """
    Find the symmetry of the chi^2 for atoms abject

    Input
        atoms:      An atom object
        sym_th:     Symmetry threshold
    Output
        sym_chi     A dictionary with independent tensor element
    """

    # Get the symmetry of the structure and operations
    import spglib
    sg = spglib.get_symmetry(atoms, symprec=sym_th)
    op_scc = sg['rotations']

    # Make a random symmterized matrix
    chi_vvv = 1 + np.random.rand(3, 3, 3)
    for v1 in range(3):
        chi_vvv[v1] = (chi_vvv[v1] + chi_vvv[v1].T) / 2.0

    # Introduce the symmetries to the matrix
    cell_cv = atoms.get_cell()
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


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (fig)
    from textwrap import wrap

    # Get the data
    data = row.data.get('results-asr.shg.json')
    if data is None:
        return

    # Make the table
    sym_chi = data.get('symm')
    table = []
    for pol in sorted(sym_chi.keys()):
        relation = sym_chi[pol]
        if pol == 'zero':
            if relation != '':
                pol = 'Others'
                relation = '0=' + relation
            else:
                continue

        if (len(relation) == 3):
            relation_new = ''
        else:
            # relation_new = '$'+'$\n$'.join(wrap(relation, 40))+'$'
            relation_new = '\n'.join(wrap(relation, 50))
        table.append((pol, relation_new))
    opt = {'type': 'table',
           'header': ['Element', 'Relations'],
           'rows': table}

    # Make the figure list
    npan = len(sym_chi)
    files = ['shg{}.png'.format(ii + 1) for ii in range(npan)]
    cols = [[fig(f'shg{2 * ii + 1}.png'),
             fig(f'shg{2 * ii + 2}.png')] for ii in range(int(npan / 2))]
    if npan % 2 == 0:
        cols.append([opt, None])
    else:
        cols.append([fig(f'shg{npan}.png'), opt])
    # Transpose the list
    cols = np.array(cols).T.tolist()

    panel = {'title': 'SHG spectrum (RPA)',
             'columns': cols,
             'plot_descriptions':
                 [{'function': plot_shg,
                   'filenames': files}],
             'sort': 20}

    return [panel]


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


@prepare_result
class Result(ASRResult):

    freqs: typing.List[float]
    chi: typing.Dict
    symm: typing.Dict
    par: typing.Dict

    key_descriptions = {
        "freqs": "Pump photon energy [eV]",
        "chi": "Non-zero SHG tensor elements in SI units",
        "symm": "Symmtery relation of SHG tensor",
        "par": "SHG paramters",
    }
    formats = {"ase_webpanel": webpanel}


@command('asr.shg',
         dependencies=['asr.structureinfo', 'asr.gs@calculate'],
         requires=['gs.gpw'],
         returns=Result)
@option(
    '--gs', help='Ground state on which response is based',
    type=str)
@option('--kptdensity', help='K-point density',
        type=float)
@option('--gauge', help='Selected gauge (length "lg" or velocity "vg")',
        type=str)
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def main(gs: str = 'gs.gpw', kptdensity: float = 20.0, gauge: str = 'lg',
         bandfactor: int = 4) -> Result:

    from ase.io import read
    from gpaw import GPAW
    from gpaw.mpi import world
    from pathlib import Path
    from gpaw.nlopt.matrixel import make_nlodata
    from gpaw.nlopt.shg import get_shg
    from gpaw.nlopt.basic import is_file

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    nd = np.sum(pbc)
    kpts = get_kpts(kptdensity, nd, atoms.get_cell())

    # SHG parameters
    eta = 0.05  # Broadening in eV
    w_ls = np.linspace(0, 10, 500)  # in eV

    try:
        # fnames = ['es.gpw', 'mml.npz']
        fnames = []
        if is_file('es.gpw'):
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

        # Calculate momentum matrix:
        mml_name = 'mml.npz'
        if is_file(mml_name):
            make_nlodata(gs_name='es.gpw', out_name=mml_name)

        # Do the calculation
        sym_chi = get_chi_symmtery(atoms)
        chi_dict = {}
        for pol in sorted(sym_chi.keys()):
            if pol == 'zero':
                continue
            # Do the SHG calculation
            shg_name = 'shg_{}.npy'.format(pol)
            if is_file(shg_name):
                shg = get_shg(
                    freqs=w_ls, eta=eta, pol=pol, gauge=gauge,
                    out_name=shg_name, mml_name=mml_name)
            else:
                shg = np.load(shg_name)

            # Make the output data
            fnames.append(shg_name)
            if nd == 2:
                chi_dict[pol] = shg[1]
            else:
                # Make it a surface chi instead of bulk chi
                cellsize = atoms.cell.cellpar()
                chi_dict[pol] = shg[1] * cellsize[2] * 1e-10

        # Make the output data
        results = {'chi': chi_dict,
                   'symm': sym_chi,
                   'freqs': w_ls,
                   'par': {'eta': eta, 'gauge': gauge,
                           'nbands': f'{(bandfactor + 1)*100}%',
                           'kpts': {'density': kptdensity, 'gamma': True}, }}

    finally:
        world.barrier()
        if world.rank == 0:
            for filename in fnames:
                es_file = Path(filename)
                if es_file.is_file():
                    es_file.unlink()

    return results


def plot_shg(row, *filename):
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    from textwrap import wrap

    # Read the data from the disk
    data = row.data.get('results-asr.shg.json')
    gap = row.get('gap')
    atoms = row.toatoms()
    pbc = atoms.pbc.tolist()
    nd = np.sum(pbc)
    if data is None:
        return

    # Remove the files if it is already exist
    for fname in filename:
        if (Path(fname).is_file()):
            os.remove(fname)

    # Plot the data and add the axis labels
    sym_chi = data['symm']
    chi = data['chi']
    w_l = data['freqs']
    fileind = 0
    axes = []
    for pol in sorted(chi.keys()):
        if pol == 'zero':
            continue

        # Make the axis and add y=0 axis
        shg = chi[pol]
        ax = plt.figure().add_subplot(111)
        ax.axhline(y=0, color='k')

        # Add the bandgap
        bg = gap
        if bg is not None:
            ax.axvline(x=bg, color='k', ls='--')
            ax.axvline(x=bg / 2, color='k', ls='--')
            maxw = min(np.ceil(2.0 * bg), 7)
        else:
            maxw = 7

        # Plot the data
        amp_l = shg * 1e18
        amp_l = amp_l[w_l < maxw]
        ax.plot(w_l[w_l < maxw], np.real(amp_l), '-', c='C0', label='Re')
        ax.plot(w_l[w_l < maxw], np.imag(amp_l), '-', c='C1', label='Im')
        ax.plot(w_l[w_l < maxw], np.abs(amp_l), '-', c='C2', label='Abs')

        # Set the axis limit
        ax.set_xlim(0, maxw)
        relation = sym_chi.get(pol)
        if not (relation is None):
            figtitle = '$' + '$\n$'.join(wrap(relation, 40)) + '$'
            ax.set_title(figtitle)
        ax.set_xlabel(r'Pump photon energy $\hbar\omega$ (eV)')
        if nd == 2:
            ax.set_ylabel(r'$\chi^{(2)}_{\gamma \alpha \beta}$ (nm$^2$/V)')
        else:
            ax.set_ylabel(r'$\chi^{(2)}_{\gamma \alpha \beta}$ (nm/V)')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

        # Add the legend
        ax.legend(loc='upper right')

        # Remove the extra space and save the figure
        plt.tight_layout()
        plt.savefig(filename[fileind])
        fileind += 1
        axes.append(ax)
        plt.close()

    # Now make the polarization resolved plot
    psi = np.linspace(0, 2 * np.pi, 201)
    selw = 0
    wind = np.argmin(np.abs(w_l - selw))
    if (Path('shgpol.npy').is_file()):
        os.remove('shgpol.npy')
    chipol = calc_polarized_shg(
        sym_chi, chi,
        wind=[wind], theta=0, phi=0,
        pte=np.sin(psi), ptm=np.cos(psi), outname=None, outbasis='pol')
    ax = plt.subplot(111, projection='polar')
    ax.plot(psi, np.abs(chipol[0]) * 1e18, 'C0', lw=1.0)
    ax.plot(psi, np.abs(chipol[1]) * 1e18, 'C1', lw=1.0)
    # Set the y limits
    ax.grid(True)
    rmax = np.amax(np.abs(chipol) * 1e18)
    if np.abs(rmax) < 1e-6:
        rmax = 1e-4
        ax.plot(0, 0, 'o', color='b', markersize=5)
    ax.set_rlim(0, 1.2 * rmax)
    ax.set_rgrids([rmax], fmt=r'%4.2g')
    labs = [r'  $\theta=0$', '45', '90', '135', '180', '225', '270', '315']
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=labs)

    # Put a legend below current axis
    ax.legend([r'Parallel: |$\chi^{(2)}_{\theta \theta \theta}$|',
               r'Perpendicular: |$\chi^{(2)}_{(\theta+90)\theta \theta}$|'],
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, ncol=2)

    # Remove the extra space and save the figure
    plt.tight_layout()
    plt.savefig(filename[fileind])
    axes.append(ax)

    return tuple(axes)


def make_full_chi(sym_chi, chi_dict):

    # Make the full chi from its symmetries
    for pol in sorted(sym_chi.keys()):
        if pol != 'zero':
            chidata = chi_dict[pol]
            nw = len(chidata)
    chi_vvvl = np.zeros((3, 3, 3, nw), complex)
    for pol in sorted(sym_chi.keys()):
        relation = sym_chi.get(pol)
        if pol == 'zero':
            if relation != '':
                for zpol in relation.split('='):
                    ind = ['xyz'.index(zpol[ii]) for ii in range(3)]
                    chi_vvvl[ind[0], ind[1], ind[2]] = np.zeros((nw), complex)
        else:
            chidata = chi_dict[pol]
            chidata = chidata[1]
            for zpol in relation.split('='):
                if zpol[0] == '-':
                    ind = ['xyz'.index(zpol[ii + 1]) for ii in range(3)]
                    chi_vvvl[ind[0], ind[1], ind[2]] = -chidata
                else:
                    ind = ['xyz'.index(zpol[ii]) for ii in range(3)]
                    chi_vvvl[ind[0], ind[1], ind[2]] = chidata

    return chi_vvvl


def calc_polarized_shg(
        sym_chi,
        chi_dict,
        wind=[1],
        theta=0.0,
        phi=0.0,
        pte=[1.0],
        ptm=[0.0],
        E0=[1.0],
        outname=None,
        outbasis='pol'):

    # Check the input arguments
    pte = np.array(pte)
    ptm = np.array(ptm)
    E0 = np.array(E0)
    assert np.all(
        np.abs(pte) ** 2 + np.abs(ptm) ** 2) == 1, \
        '|pte|**2+|ptm|**2 should be one.'
    assert len(pte) == len(ptm), 'Size of pte and ptm should be the same.'

    # Useful variables
    costh = np.cos(theta)
    sinth = np.sin(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    nw = len(wind)
    npsi = len(pte)

    # Transfer matrix between (x y z)/(atm ate k) unit vectors basis
    if theta == 0:
        transmat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        transmat = [[cosphi * costh, sinphi * costh, -sinth],
                    [-sinphi, cosphi, 0],
                    [sinth * cosphi, sinth * sinphi, costh]]
    transmat = np.array(transmat)

    # Get the full chi tensor
    chi_vvvl = make_full_chi(sym_chi, chi_dict)

    # Check the E0
    if len(E0) == 1:
        E0 = E0 * np.ones((nw))

    # in xyz coordinate
    Einc = np.zeros((3, npsi), dtype=complex)
    for v1 in range(3):
        Einc[v1] = (pte * transmat[0][v1] + ptm * transmat[1][v1])

    # Loop over components
    chipol = np.zeros((3, npsi, nw), dtype=complex)
    for ii, wi in enumerate(wind):
        for ind in range(27):
            v1, v2, v3 = int(ind / 9), int((ind % 9) / 3), (ind % 9) % 3
            if chi_vvvl[v1, v2, v3, wi] != 0.0:
                chipol[v1, :, ii] += chi_vvvl[v1, v2, v3, wi] * \
                    Einc[v2, :] * Einc[v3, :] * E0[ii]**2

    # Change the output basis if needed, and return
    if outbasis == 'xyz':
        chipol_new = chipol
    elif outbasis == 'pol':
        chipol_new = np.zeros((3, npsi, nw), dtype=complex)
        for ind, wi in enumerate(wind):
            chipol[:, :, ind] = np.dot(transmat.T, chipol[:, :, ind])
            chipol_new[0, :, ind] = chipol[0, :, ind] * \
                pte + chipol[1, :, ind] * ptm
            chipol_new[1, :, ind] = -chipol[0, :, ind] * \
                ptm + chipol[1, :, ind] * pte

    else:
        raise NotImplementedError

    # Save it to the file
    if outname is None:
        np.save('shgpol.npy', chipol_new)
    else:
        np.save('{}.npy'.format(outname), chipol_new)
    return chipol_new


if __name__ == '__main__':
    main.cli()
