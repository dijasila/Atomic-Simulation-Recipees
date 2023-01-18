from asr.core import command, option, ASRResult, prepare_result
from typing import Union
import numpy as np


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig
    spiraltable = table(row, 'Property', ['bandwidth', 'minimum'], key_descriptions)
    panel = {'title': 'Spin spirals',
             'columns': [[fig('spin_spiral_bs.png')], [spiraltable]],
             'plot_descriptions': [{'function': plot_bandstructure,
                                    'filenames': ['spin_spiral_bs.png']}],
             'sort': 3}
    return [panel]


@prepare_result
class Result(ASRResult):
    path: np.ndarray
    energies: np.ndarray
    local_magmoms: np.ndarray
    total_magmoms: np.ndarray
    bandwidth: float
    minimum: np.ndarray
    key_descriptions = {"path": "List of Spin spiral vectors",
                        "energies": "Potential energy [eV]",
                        "local_magmoms": "List of estimated local moments [mu_B]",
                        "total_magmoms": "Estimated total moment [mu_B]",
                        "bandwidth": "Energy difference [meV]",
                        "minimum": "Q-vector at energy minimum"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.spinspiral',
         requires=['structure.json'],
         # dependencies=['asr.spinspiral@calculate'],
         returns=Result)
@option('--q_path', help='Spin spiral high symmetry path eg. "GKMG"', type=str)
@option('--n', type=int)
@option('--params', help='Calculator parameter dictionary', type=dict)
@option('--eps', help='Bandpath symmetry threshold', type=float)
def main(q_path: Union[str, None] = None, n: int = 11,
         params: dict = dict(mode={'name': 'pw', 'ecut': 600},
                             kpts={'density': 6.0, 'gamma': True}),
         eps: float = 0.0002) -> Result:
    from ase.io import read
    import json
    from glob import glob
    atoms = read('structure.json')
    c2db_eps = 0.1
    path = atoms.cell.bandpath(npoints=0, pbc=atoms.pbc, eps=c2db_eps)
    Q = path.kpts

    data = []
    jsons = glob('dat*.json')
    for js in jsons:
        print(f'Collecting {js}')
        with open(js, 'r') as fd:
            data_s = json.load(fd)
            data.append(data_s)

    # Extract strings from different orders calculated basede on output jsons
    orders = list(set([list(i.keys())[0] for i in data]))
    keys = ['n', 'e', 'm_v', 'm_av']

    # append data to results dictionary
    results = {order: {k: [] for k in keys} for order in orders}
    for this_data in data:
        for order in this_data.keys():
            for key in this_data[order].keys():
                results[order][key].append(this_data[order][key])

    res = {}
    for order in orders:
        n = results[order]['n']
        for key in keys:
            # sort data based on the counter "n"
            res_i = results[order][key]
            _, res_i = zip(*sorted(zip(n, res_i)))

            # restructure output
            if key in res.keys():
                res[key] = np.append(res[key], np.asarray([res_i]), axis=0)
            else:
                res[key] = np.array([np.asarray(res_i)])

        if 'order' in res.keys():
            res['order'] = np.append(res['order'], np.asarray([order]), axis=0)
        else:
            res['order'] = np.asarray([order])

    bandwidth = (np.max(res['e']) - np.min(res['e'])) * 1000
    minarg = np.unravel_index(np.argmin(res['e'], axis=None), res['e'].shape)
    omin = orders[minarg[0]]
    qmin = Q[minarg[1]]
    return Result.fromdata(path=path, energies=res['e'],
                           local_magmoms=res['m_av'], total_magmoms=res['m_v'],
                           bandwidth=bandwidth, minimum=(qmin, omin))


def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    data = row.data.get('results-asr.spinspiral.json')
    path = data['path']
    q, x, X = path.get_linear_kpoint_axis()
    energies_o = data['energies']
    if len(energies_o.shape) == 1:
        energies_o = np.array([energies_o])
        local_magmoms_o = np.array([data['local_magmoms']])
    else:
        local_magmoms_o = data['local_magmoms']

    mommin = np.min(abs(local_magmoms_o) * 0.9)
    mommax = np.max(abs(local_magmoms_o) * 1.05)

    e0 = energies_o.flatten()[0]
    emin = np.min(1000 * (energies_o - e0) * 1.1)
    emax = np.max(1000 * (energies_o - e0) * 1.15)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for o in range(energies_o.shape[0]):
        energies = energies_o[o]
        energies = ((energies - e0) * 1000)  # / nmagatoms
        local_magmoms = local_magmoms_o[o]

        # Setup main energy plot
        hnd = ax1.plot(q, energies, c='C0', marker='.', label='Energy')
        ax1.set_ylim([emin, emax])
        ax1.set_xticks(x)
        ax1.set_xticklabels([i.replace('G', r"$\Gamma$") for i in X])
        for xc in x:
            if xc != min(q) and xc != max(q):
                ax1.axvline(xc, c='gray', linestyle='--')
        ax1.margins(x=0)

        # Add spin wavelength axis
        def tick_function(X):
            lmda = 2 * np.pi / X
            return [f"{z:.1f}" for z in lmda]

        # Non-cumulative length of q-vectors to find wavelength
        Q = np.linalg.norm(2 * np.pi * path.cartesian_kpts(), axis=-1)
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        idx = round(len(Q) / 5)
        ax2.set_xticks(q[::idx])
        ax2.set_xticklabels(tick_function(Q[::idx]))

        # Add the magnetic moment plot
        ax3 = ax1.twinx()
        try:
            row = getattr(row, '_row')
            s = row['symbols']
        except AttributeError:
            s = row.symbols

        unique = list(set(s))
        colors = [f'C{i}' for i in range(1, len(unique) + 1)]
        mag_c = {unique[i]: colors[i] for i in range(len(unique))}

        magmom_qa = np.linalg.norm(local_magmoms, axis=2)
        for a in range(magmom_qa.shape[-1]):
            magmom_q = magmom_qa[:, a]
            ax3.plot(q, magmom_q, c=mag_c[s[a]], marker='.', label=f'{s[a]} magmom')

        ax3.set_ylim([mommin, mommax])

        # Ensure unique legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        updict = {'Energy': hnd[0]}
        updict.update(by_label)
        fig.legend(updict.values(), updict.keys(), loc="upper right",
                   bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    ax1.set_title(str(row.formula), fontsize=14)
    ax1.set_ylabel('Spin spiral energy [meV]')
    ax1.set_xlabel('q vector [Å$^{-1}$]')

    ax2.set_xlabel(r"Wave length $\lambda$ [Å]")
    ax3.set_ylabel(r"Local norm magnetic moment [|$\mu_B$|]")

    # fig.suptitle('')
    plt.tight_layout()
    plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
