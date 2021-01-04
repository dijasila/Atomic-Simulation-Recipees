import typing

from asr.core import command, option, ASRResult, prepare_result
from asr.shg import CentroSymmetric, get_chi_symmetry, get_kpts
import numpy as np


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (fig)
    from textwrap import wrap

    # Get the data
    data = row.data.get('results-asr.shift.json')

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
    npan = len(sym_chi) - 1
    files = ['shift{}.png'.format(ii + 1) for ii in range(npan)]
    cols = [[fig(f'shift{2 * ii + 1}.png'),
             fig(f'shift{2 * ii + 2}.png')] for ii in range(int(npan / 2))]
    if npan % 2 == 0:
        cols.append([opt, None])
    else:
        cols.append([fig(f'shift{npan}.png'), opt])
    # Transpose the list
    cols = np.array(cols).T.tolist()

    panel = {'title': 'Shift current spectrum (RPA)',
             'columns': cols,
             'plot_descriptions':
                 [{'function': plot_shift,
                   'filenames': files}],
             'sort': 20}

    return [panel]


@prepare_result
class Result(ASRResult):

    freqs: typing.List[float]
    sigma: typing.Dict[str, typing.Any]
    symm: typing.Dict[str, str]

    key_descriptions = {
        "freqs": "Pump photon energy [eV]",
        "sigma": "Non-zero shift conductivity tensor elements in SI units",
        "symm": "Symmetry relation of shift conductivity tensor",
    }
    formats = {"ase_webpanel": webpanel}


@command('asr.shift',
         dependencies=['asr.gs@calculate'],
         requires=['structure.json', 'gs.gpw'],
         returns=Result)
@option('--gs', help='Ground state on which response is based',
        type=str)
@option('--kptdensity', help='K-point density', type=float)
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
@option('--eta', help='Broadening [eV]', type=float)
@option('--maxomega', help='Max pump frequency [eV]', type=float)
@option('--nromega', help='Number of pump frequencies', type=int)
@option('--energytol', help='Energy tolernce [eV]', type=float)
@option('--removefiles', help='Remove created files', type=bool)
def main(gs: str = 'gs.gpw', kptdensity: float = 25.0,
         bandfactor: int = 4, eta: float = 0.05, energytol: float = 1e-6,
         maxomega: float = 10.0, nromega: int = 1000,
         removefiles: bool = False) -> Result:
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
        Energy tolernce to remove degeneracies.
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


def plot_shift(row, *filename):
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    from textwrap import wrap

    # Read the data from the disk
    data = row.data.get('results-asr.shift.json')
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
    if len(sym_chi) == 1:
        raise CentroSymmetric
    sigma = data['sigma']

    if not sigma:
        return
    w_l = data['freqs']
    fileind = 0
    axes = []

    for pol in sorted(sigma.keys()):
        # Make the axis and add y=0 axis
        shift = sigma[pol]
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
        amp_l = shift
        amp_l = amp_l[w_l < maxw]
        ax.plot(w_l[w_l < maxw], np.real(amp_l), '-', c='C0', label='Re')

        # Set the axis limit
        ax.set_xlim(0, maxw)
        relation = sym_chi.get(pol)
        if not (relation is None):
            figtitle = '$' + '$\n$'.join(wrap(relation, 40)) + '$'
            ax.set_title(figtitle)
        ax.set_xlabel(r'Pump photon energy $\hbar\omega$ [eV]')
        polstr = f'{pol}'
        if nd == 2:
            ax.set_ylabel(r'$\sigma^{(2)}_{' + polstr + r'}$ [nm$\mu$A/V$^2$]')
        else:
            ax.set_ylabel(r'$\sigma^{(2)}_{' + polstr + r'} [$\mu$A/V$^2$]')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

        # Remove the extra space and save the figure
        plt.tight_layout()
        plt.savefig(filename[fileind])
        fileind += 1
        axes.append(ax)
        plt.close()

    return tuple(axes)


if __name__ == '__main__':
    main.cli()
