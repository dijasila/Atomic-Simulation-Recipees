import typing
from asr.core import command, option, ASRResult, prepare_result
import numpy as np


def find_zero_element(atoms, sym_th=1e-4):

    from ase.spacegroup import get_spacegroup

    cell_cv = atoms.get_cell()
    cell_vc = np.linalg.inv(cell_cv)
    sg = get_spacegroup(atoms, symprec=1e-3)
    randchi = np.random.rand(3, 3, 3)
    rots = sg.rotations

    # Loop over symmetries
    zero_ind = []
    for rot_cc in rots:
        rot = np.dot(cell_vc, np.dot(rot_cc.T, cell_cv))

        randchi2 = np.einsum('il,jm,kn,lmn->ijk', rot, rot, rot, randchi)
        ind = np.where(np.abs(randchi2 + randchi) < sym_th)
        if ind[0].size > 0:
            for ii, jj, kk in zip(ind[0], ind[1], ind[2]):
                zero_ind.append('xyz'[ii] + 'xyz'[jj] + 'xyz'[kk])

    return zero_ind


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig

    # Get the data
    data = row.data.get('results-asr.shg.json')
    if data is None:
        return

    # Make the figure list
    pols = data.get('pols')
    nrelement = len(pols)
    files = ['shg{}.png'.format(ii + 1) for ii in range(nrelement)]
    if nrelement % 2 == 0:
        cols = [[fig('shg{}.png'.format(2 * ii + 1)),
                 fig('shg{}.png'.format(2 * ii + 2))]
                for ii in range(int(nrelement / 2))]
    else:
        cols = [[fig('shg{}.png'.format(2 * ii + 1)),
                 fig('shg{}.png'.format(2 * ii + 2))]
                for ii in range(int(nrelement / 2))]
        cols.append([fig('shg{}.png'.format(nrelement)), None])
    # Transpose the list
    cols = np.array(cols).T.tolist()
    panel = {'title': 'SHG spectrum (RPA)',
             'columns': cols,
             'plot_descriptions':
                 [{'function': shg,
                   'filenames': files}],
             'sort': 20}

    return [panel]


def get_kpts(kptdensity, pbc, cell):
    kpts = {}
    ND = np.sum(pbc)
    if ND == 3 or ND == 1:
        kpts = {'density': kptdensity, 'gamma': False, 'even': True}
    elif ND == 2:
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

    eta: float
    freqs_l: typing.List[float]
    shg_vvvl: typing.List[typing.List[typing.List[typing.List[complex]]]]

    key_descriptions = {
        "eta": "Broadening [eV]",
        "freqs_l": "Pump photon frequencies [eV]",
        "shg_vvvl": "SHG tensor [m/V or m$^2$/V]",
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
@option('--ecut', help='Plane wave cutoff',
        type=float)
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def main(gs: str = 'gs.gpw', kptdensity: float = 20.0, ecut: float = 50.0,
         bandfactor: int = 4) -> Result:

    from ase.io import read
    from gpaw import GPAW
    # from gpaw.mpi import world
    from pathlib import Path
    from gpaw.nlopt.matrixel import make_nlodata
    from gpaw.nlopt.shg import get_shg
    from gpaw.nlopt.basic import is_file

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    kpts = get_kpts(kptdensity, pbc, atoms.get_cell())

    try:
        if not Path('es.gpw').is_file():
            calc_old = GPAW(gs, txt=None)
            nval = calc_old.wfs.nvalence

            calc = GPAW(
                gs,
                txt='es.txt',
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
        eta = 0.05  # Broadening in eV
        w_ls = np.linspace(0, 10, 500)  # in eV
        shg_vvvl = np.zeros((3, 3, 3, len(w_ls)), complex)
        data = {}
        pols = []
        zero_ind = find_zero_element(atoms)
        for ind in range(27):
            v1, v2, v3 = int(ind / 9), int(int(ind % 9) / 3), int(ind % 9) % 3
            pol = 'xyz'[v1] + 'xyz'[v2] + 'xyz'[v3]
            # The tensor is symmetric with respect to the last two indices
            if v3 < v2 or (pol in zero_ind):
                continue

            # SHG calculation
            pols.append(pol)
            shg_name = 'shg_{}.npy'.format(pol)
            if is_file(shg_name):
                shg = get_shg(
                    freqs=w_ls, eta=eta, pol=pol, gauge='lg',
                    out_name=shg_name, mml_name=mml_name)
            else:
                shg = np.load(shg_name)

            # Make the output data
            shg_vvvl[v1, v2, v3] = shg[1]
            shg_vvvl[v1, v3, v2] = shg[1]

        # Make the output data
        data['shg_vvvl'] = shg_vvvl
        data['freq_l'] = w_ls
        data['eta'] = eta
        data['pols'] = pols

    finally:
        pass

    return data


# Make the graphs
def shg(row, *filename):
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path

    # Read the data from the disk
    data = row.data.get('results-asr.shg.json')
    gap = row.get('gap')
    if data is None:
        return
    shg_vvvl = data['shg_vvvl']
    w_l = data['freq_l']
    atoms = row.toatoms()

    # Remove the files if it is already exist
    for fname in filename:
        if (Path(fname).is_file()):
            os.remove(fname)

    # Plot the data and add the axis labels
    zero_ind = find_zero_element(atoms)
    fileind = 0
    axes = []
    for ind in range(27):
        v1, v2, v3 = int(ind / 9), int(int(ind % 9) / 3), int(ind % 9) % 3
        pol = 'xyz'[v1] + 'xyz'[v2] + 'xyz'[v3]
        # The tensor is symmetric with respect to the last two indices
        if v3 < v2 or (pol in zero_ind):
            continue

        # Make the axis and add y=0 axis
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
        amp_l = shg_vvvl[v1, v2, v3] * 1e18
        amp_l = amp_l[w_l < maxw]
        ax.plot(w_l[w_l < maxw], np.real(amp_l), '-', c='C0', label='Re')
        ax.plot(w_l[w_l < maxw], np.imag(amp_l), '-', c='C1', label='Im')
        ax.plot(w_l[w_l < maxw], np.abs(amp_l), '-', c='C2', label='Abs')

        # Set the axis limit
        ax.set_xlim(0, maxw)

        # Add the title and labels
        ax.set_xlabel(r'Pump photon energy $\hbar\omega$ (eV)')
        ax.set_ylabel(r'$\chi^{(2)}_{\gamma \alpha \beta}$ (nm$^2$/V)')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

        # Add the legend
        ax.legend(loc='upper right')

        # Remove the extra space and save the figure
        plt.tight_layout()
        plt.savefig(filename[fileind])
        fileind += 1

        # Save the axis and close the figure
        axes.append(ax)
        # plt.close()

    return tuple(axes)


if __name__ == '__main__':
    main.cli()
