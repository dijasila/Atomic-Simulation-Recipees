from dataclasses import dataclass
from asr.core import command, option, ASRResult, prepare_result
from typing import List
import numpy as np
import json


@dataclass
class SpinSpiralCalculation:
    index: List[int]         # list of [band, qpt] of this calculation
    energy: float            # total energy of this calculation
    m_v: List[float]         # total magnetic moments in Cartesian directions
    m_av: List[List[float]]  # magmoms resolved by atom and direction
    gap: float               # Bandgap of this calculation

    def save(self, filename):
        with open(filename, 'w') as fd:
            json.dump(dict(index=self.index, energy=self.energy,
                           m_v=self.m_v, m_av=self.m_av, gap=self.gap), fd)

    @classmethod
    def load(self, filename):
        with open(filename, 'r') as fd:
            data = json.load(fd)
        return SpinSpiralCalculation(**data)


class SpinSpiralPathCalculation:
    def __init__(self):
        self.sscalculations: List[SpinSpiralCalculation] = []
        self.indices = []

    def __iter__(self):
        return iter(self.sscalculations)

    def __repr__(self):
        return self.sscalculations

    def __getitem__(self, item):
        return self.sscalculations[item]

    def append(self, sscalc: SpinSpiralCalculation):
        self.sscalculations.append(sscalc)
        self.indices.append(sscalc.index)

    def sort_path(self):
        self.sscalculations, self.indices = zip(
            *sorted(zip(self.sscalculations, self.indices),
                    key=lambda x: (x[1][0], x[1][1])))
        self.indices = list(self.indices)

    def get_idx(self, index):
        return [sscalc for sscalc in self.sscalculations if sscalc.index == index]

    def failed_calculations(self, nkpts: int):
        assert len(self.indices) > 0
        natoms = len(self.sscalculations[0].m_av)
        bands = [i[0] for i in self.indices]
        for iband in bands:
            for ikpt in range(nkpts):
                # Yield zero calculation if index combination [band, qpt] isn't found
                if len(self.get_idx([iband, ikpt])) == 0:
                    zerospiral = SpinSpiralCalculation([iband, ikpt], 0.0, [0, 0, 0],
                                                       [[0, 0, 0]] * natoms, 0)
                    yield zerospiral

    def get_array(self, attr: str = 'energy'):
        Nbands, Nqpts = max(self.indices)
        array = np.asarray([[getattr(self.get_idx([ib, iq])[0], attr)
                             for iq in range(Nqpts + 1)] for ib in range(Nbands + 1)])
        return array


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
    gaps: np.ndarray
    local_magmoms: np.ndarray
    total_magmoms: np.ndarray
    bandwidth: float
    minimum: list
    Qmin: np.ndarray
    key_descriptions = {"path": "List of Spin spiral vectors",
                        "energies": "Total energy of spin spiral calculations [eV]",
                        "gaps": "Bandgaps of spin spiral calculations [eV]",
                        "local_magmoms": "Estimated local moments [mu_B]",
                        "total_magmoms": "Estimated total moment [mu_B]",
                        "bandwidth": "Energy bandwidth [meV]",
                        "minimum": "Band and qpt index of energy minimum",
                        "Qmin": "Q-vector of energy minimum in fractional coordinates"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.collect_spiral',
         requires=['structure.json'],
         returns=Result)
@option('--qdens', type=int)
def main(qdens: int = 22) -> Result:
    from ase.io import read
    from glob import glob
    atoms = read('structure.json')
    jsons = glob('dat*.json')

    sscalcs = SpinSpiralPathCalculation()
    for js in jsons:
        sscalc = SpinSpiralCalculation.load(js)
        sscalcs.append(sscalc)

    return _main(atoms, sscalcs, qdens)


def _main(atoms, sscalculations, qdens):
    c2db_eps = 0.1
    path = atoms.cell.bandpath(density=qdens, pbc=atoms.pbc, eps=c2db_eps)
    Q = path.kpts
    nqpts = len(Q)

    energies = [sscalc.energy for sscalc in sscalculations]
    minarg = np.argmin(energies)
    min_sscalc = sscalculations[minarg]
    minimum = min_sscalc.index
    Qmin = Q[minimum[1]]

    bandwidth = (np.max(energies) - np.min(energies)) * 1000

    for zerospiral in sscalculations.failed_calculations(nqpts):
        sscalculations.append(zerospiral)

    sscalculations.sort_path()
    energies = sscalculations.get_array('energy')
    m_v = sscalculations.get_array('m_v')
    m_av = sscalculations.get_array('m_av')
    gaps = sscalculations.get_array('gap')

    return Result.fromdata(path=path, energies=energies, gaps=gaps,
                           local_magmoms=m_av, total_magmoms=m_v,
                           bandwidth=bandwidth, minimum=minimum,
                           Qmin=Qmin)


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
