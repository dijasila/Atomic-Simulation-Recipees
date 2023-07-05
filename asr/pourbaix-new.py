import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from itertools import product, chain, combinations, permutations
from sympy import Matrix
from ase.units import kB
from ase.formula import Formula


CONST = kB * np.log(10)

PREDEF_ENERGIES = {
    'H+': 0.0,
    'e-': 0.0,
    'O': -4.57,     # http://dx.doi.org/10.1103/PhysRevB.85.235438
    'H': -3.73,     # http://dx.doi.org/10.1103/PhysRevB.85.235438
    'H2O': -2.4583
}


def initialize_refs(refs_dct):
    """Convert dictionary entries to Species instances"""
    refs = {}
    for name, energy in refs_dct.items():
        spec = Species(name)
        spec.set_chemical_potential(energy, None)
        refs[name] = spec
    return refs


def get_solid_refs(material, db_name):
    """Extract solid references in the compositional space
       of a given material from a database"""
    from ase.db import connect

    db = connect(db_name)
    element_energies = {}
    refs = {}

    for subsys in material.get_chemsys():
        nspecies = len(subsys)
        query_str = ",".join(subsys) + f',nspecies={nspecies}'

        for row in db.select(query_str):
            energy = row.energy
            ref = Species(row.formula)
            name = ref.name

            if nspecies == 1:
                energy_elem = PREDEF_ENERGIES.get(
                    name,
                    energy / row.natoms
                )
                element_energies[name] = energy_elem

            OH_content = ref.get_fractional_composition(['O', 'H'])
            if OH_content > 0.85 or 1e-4 < OH_content < 0.095:
                continue

            chempot = ref.get_formation_energy(energy, element_energies)
            refs[name] = chempot

    return refs, element_energies


def get_solvated_refs(name):
    """Extract ionic and solvated species"""
    from ase.phasediagram import solvated
    ref_dct = {}
    solv = solvated(name)
    for name, energy in solv:
        if name not in ['H+(aq)', 'H2O(aq)']:
            ref_dct[name] = energy
    return ref_dct


def get_references(material, db_name, computed_energy=None, include_aq=True):
    from ase.phasediagram import solvated

    species = Species(material)
    refs, element_energies = get_solid_refs(species, db_name)
    refs.update(get_solvated_refs(species.name))

    if computed_energy:
        chempot = species.get_formation_energy(
            computed_energy, element_energies
        )
        refs[species.name] = chempot
    else:
        if species.name not in refs.keys():
            raise ValueError(\
                f'Your material has not been found in {db_name}. '
                f'Please provide a total energy for it!')
    return refs


def get_product_combos(reactant, refs):
    array = [[] for i in range(len(reactant.elements))]
    for ref in refs.values():
        contained = ref.contains(reactant.elements)
        for w in np.argwhere(contained).flatten():
            array[w].append(ref)
    return product(*array)


def get_phases(reactant, refs, T, conc, counter, normalize=True):
    """Obtain all the possible decomposition pathways
       for a given reactant."""

    # Initialize phases with the special case where our reactant is stable
    reactant_stable = NoReaction(reactant)
    phases = [reactant_stable]
    phase_matrix = [reactant_stable._vector]
    reac_elem = [-reactant._count_array(reactant.elements)]

    for products in get_product_combos(reactant, refs):
        if len(np.unique(products)) < len(products):
            continue

        prod_elem = [p._count_array(reactant.elements) for p in products]
        elem_matrix = np.array(reac_elem + prod_elem).T
        solutions = Matrix(elem_matrix).nullspace()

        for solution in solutions:
            coeffs = np.array(solution).flatten()
            if all(coeffs > 0):
                if normalize:
                    coeffs /= abs(coeffs[0])
                coeffs[0] = -coeffs[0]
                species = (reactant, *products)
                phase = RedOx(species, coeffs, T, conc, counter)
                phases.append(phase)
                phase_matrix.append(phase._vector)

    return phases, np.array(phase_matrix).astype('float64')


def edge_detection(array):
    from collections import defaultdict
    edges_raw = defaultdict(list)
    edges = defaultdict(list)

    for i in range(array.shape[0] - 1):
        for j in range(array.shape[1] - 1):
            xpair = (array[i, j], array[i+1, j])
            ypair = (array[i, j], array[i, j+1])
            for pair in [xpair, ypair]:
                if np.ptp(pair) != 0:
                    edges_raw[pair].append([i, j])

    for pair, values in edges_raw.items():
        varr = np.array(values)
        left = values[np.argmin(varr[:, 1])]
        right = values[np.argmax(varr[:, 1])]
        edges[pair] = np.array([left, right]).T

    return edges


def add_numbers(ax, text):
    import matplotlib.patheffects as pfx
    for i, (x, y, prod) in enumerate(text):
        txt = ax.text(
            y, x, f'{i}', 
            fontsize=20,
            horizontalalignment='center'
        )
        txt.set_path_effects([pfx.withStroke(linewidth=2.0, foreground='w')])
    return

def add_text(ax, text, offset=0):
    # Adding text on the diagram
    import textwrap
    import re

    textlines = []
    for i, (x, y, prod) in enumerate(text):
        formatted = []
        for p in prod:
        #label = ', '.join(p for p in prod
            label = re.sub(r'(\S)([+-]+)', r'\1$^{\2}$', p)
            label = re.sub(r'(\d+)', r'$_{\1}$', label)
            for symbol in ['+', '-']:
                count = label.count('+')
                if count > 1:
                    label = label.replace(count*symbol, f'{count}{symbol}')
                if count == 1:
                    label = label.replace(count*symbol, symbol)
            formatted.append(label)

        label = ', '.join(f for f in formatted)
        textlines.append(
            textwrap.fill(f'({i})  {label}', 
                          width=40,
                          subsequent_indent='      ')
        )
    text = "\n".join(textlines)
    plt.gcf().text(
            0.75 + offset, 0.5,
            text,
            fontsize=16,
            va='center',
            ha='left')
    return 0


def add_redox_lines(axes, pH, color='k'):
    # Add water redox potentials
    slope = -59.2e-3
    axes.plot(pH, slope*pH, c=color, ls='--', zorder=2)
    axes.plot(pH, slope*pH + 1.229, c=color, ls='--', zorder=2)
    return 0


class Species:
    '''
    Groups relevant quantities for a single chemical species
    '''
    def __init__(self, formula, fmt='metal'):
        self.aq = formula.endswith('(aq)')
        formula_strip = formula.replace('(aq)', '').rstrip('+-')
        self.charge = formula.count('+') - formula.count('-')
        formula_obj = Formula(formula_strip, format=fmt)
        self._count = formula_obj.count()

        if self.aq:
            self.name = formula
            self.n_fu = 1
            self.count = self._count
        else:
            reduced, self.n_fu = formula_obj.reduce()
            self.count = reduced.count()
            self.name = str(reduced)

        self._elements = [elem for elem in self.count]
        self.elements = [elem for elem in self._elements if elem not in ['H', 'O']]
        self.natoms = sum(self.count.values())
        self.energy = None
        self.mu = None

    def get_chemsys(self):
        elements = set(self.count.keys())
        elements.update(['H', 'O'])
        chemsys = list(
            chain.from_iterable(
                [combinations(elements, i+1) for i,_ in enumerate(list(elements))]
            )
        )
        return chemsys

    def balance_electrochemistry(self):
        '''Obtain number of H2O, H+, e- "carried" by the species'''

        n_H2O = -self.count.get('O', 0)
        n_H = -2 * n_H2O - self.count.get('H', 0)
        n_e = n_H + self.charge
        return n_H2O, n_H, n_e

    def _count_array(self, elements):
        return np.array([self.count.get(e, 0) for e in elements])

    def contains(self, elements):
        return [True if elem in self.elements else False for elem in elements]

    def get_fractional_composition(self, elements):
        N_all = sum(self.count.values())
        N_elem = sum([self.count.get(e, 0) for e in elements])
        return N_elem / N_all

    def get_formation_energy(self, energy, refs):
        elem_energy = sum([refs[s] * n for s, n in self._count.items()])
        hof = (energy - elem_energy) / self.n_fu
        return hof

    def set_chemical_potential(self, energy, refs=None):
        self.energy = energy
        if refs is None:
            self.mu = energy / self.n_fu
        else:
            self.mu = self.get_formation_energy(energy, refs)

    def __repr__(self):
        return f'({self.name}, μ={self.mu})'


class RedOx:
    def __init__(self, species, coeffs,
                 T=298.15, conc=1e-6,
                 counter='SHE'):

        alpha = CONST * T   # 0.059 eV @ T=298.15K
        const_term = 0
        pH_term = 0
        U_term = 0
        self.species = Counter()

        for spec, coef in zip(species, coeffs):
            self.species[spec.name] = coef
            amounts = spec.balance_electrochemistry()

            const_term += coef * ( \
                spec.mu + alpha * (spec.aq * np.log10(conc)))
            pH_term += - coef * alpha * amounts[1]
            U_term += - coef * amounts[2]

            for name, n in zip(['H2O', 'H+', 'e-'], amounts):
                const_term += coef * n * PREDEF_ENERGIES[name]
                self.species[name] += coef * n

        const_corr, pH_corr = self.get_counter_correction(counter, alpha)
        self._vector = [
            float(const_term + const_corr), 
            float(U_term), 
            float(pH_term + pH_corr)
        ]

    def get_counter_correction(self, counter, alpha):
        n_e = self.species['e-']
        gibbs_corr = 0.0
        pH_corr = 0.0
        if counter in ['RHE', 'Pt']:
            pH_corr += n_e * alpha
            if counter == 'Pt' and n_e < 0:
                gibbs_corr +=  n_e * 0.5 * PREDEF_ENERGIES['H2O']
        return gibbs_corr, pH_corr

    def equation(self):
        reactants = []
        products = []
        for s, n in self.species.items():
            if n == 0:
                continue
            if abs(n) == 1:
                substr = s
            else:
                substr = f'{abs(n)}{s}'
            if n > 0:
                products.append(substr)
            else:
                reactants.append(substr)
        return "  ➜  ".join([" + ".join(reactants), " + ".join(products)])

    def get_main_products(self):
        return [spec for spec, coef in self.species.items() 
                if coef > 0 and spec not in ['H+', 'H2O', 'e-']]


class NoReaction(RedOx):
    def __init__(self, reactant):
        self.name = reactant.name
        self.species = {reactant.name: 1}
        self._vector = [0.0, 0.0, 0.0]

    def equation(self):
        return self.name


class Pourbaix:
    def __init__(self, material_name, refs_dct, T=298.15, conc=1.0e-6, counter='SHE'):
        refs = initialize_refs(refs_dct)
        material = refs.pop(material_name)
        self.natoms = material.natoms
        self.counter = counter

        self.phases, phase_matrix = get_phases(
            material, refs, T, conc, counter
        )
        self._const = phase_matrix[:, 0]
        self._var = phase_matrix[:, 1:]

    def _decompose(self, U, pH):
        '''Evaluate the reaction energy for decomposing
           the target material into the most stable phase
           at a given pH and applied potential.

           if zero, the target material IS the most stable phase.
        '''
        return self._const + np.dot(self._var, [U, pH])

    def _get_pourbaix_energy(self, U, pH):
        '''Evaluate the energy of the target material
           relative to the most stable phase
        '''
        energies = self._decompose(U, pH)
        i_min = np.argmin(energies)
        return -energies[i_min], i_min

    def get_pourbaix_energy(self, U, pH, verbose=True):
        '''Evaluate Pourbaix energy and obtain info
           about the most stable phase
        '''
        energy, index = self._get_pourbaix_energy(U, pH)
        phase = self.phases[index]
        if verbose:
            print(f'Stable phase: \n{phase.equation()}'
                  f'\nEnergy: {energy} eV')
        return energy, phase

    def get_diagrams(self, U, pH):

        pour = np.zeros((len(U), len(pH)))
        meta = pour.copy()

        for i, u in enumerate(U):
            for j, p in enumerate(pH):
                meta[i, j], pour[i, j] = self._get_pourbaix_energy(u, p)

        text = []
        for phase_id in np.unique(pour):
            phase = self.phases[int(phase_id)]
            where = (pour == phase_id)
            x = np.dot(where.sum(1), U) / where.sum()
            y = np.dot(where.sum(0), pH) / where.sum()
            text.append((x, y, phase.get_main_products()))

        return pour, meta, text

    def draw_diagram_axes(
            self,
            Urange, pHrange,
            npoints=300,
            normalize=True,
            cap=1.0,
            figsize=[12, 6],
            cmap="RdYlGn_r"):

        ratio = np.ptp(Urange) / np.ptp(pHrange)
        ratio = figsize[1] / figsize[0]
        pH = np.linspace(*pHrange, num=npoints, retstep=True)
        U = np.linspace(*Urange, num=int(npoints))

        pour, meta, text = self.get_diagrams(U, pH)
        levels = np.unique(pour)

        if normalize:
            meta /= self.natoms

        ax = plt.figure(figsize=figsize).add_subplot(111)
        extent = [*pHrange, *Urange]

        plt.subplots_adjust(
            left=0.1, right=0.75,
            top=0.97, bottom=0.14
        )

        colorplot = ax.imshow(
            meta, cmap=cmap,
            extent=extent, vmax=cap,
            origin='lower', aspect='auto',
            interpolation='gaussian'
        )
            
        cbar = plt.gcf().colorbar(
               colorplot,
               ax=ax,
               pad=0.02
        )

        ax.set_xlabel('pH', fontsize=18)
        ax.set_ylabel(f'Potential vs. {self.counter} (V)', fontsize=18)
        ax.set_xlim(*pHrange)
        ax.set_ylim(*Urange)
        ax.set_xticks(np.arange(pHrange[0], pHrange[1] + 1, 2))
        ax.set_yticks(np.arange(Urange[0], Urange[1] + 1, 1))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ticks = np.linspace(0, cap, num=6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_ylabel(r'$E_{pbx}$ (eV/atom)', fontsize=18)

        edges = edge_detection(pour)
        for _, indexes in edges.items():
            ax.plot(
                pH[indexes[1]],
                U[indexes[0]],
                ls='-',
                marker=None,
                zorder=1,
                color='k'
            )

        add_numbers(ax, text)
        add_text(ax, text, offset=0.05)
        add_redox_lines(ax, pH, 'w')
        plt.show()

        return ax


    '''
    def plot(self, U, pH, 
             cmap="RdYlGn",
             cap=1.0,
             normalize=True,
             savefig='pourbaix.png',
             figsize=[12, 6],
             show=True):

        ax = draw_diagram_axes(pour, meta, text, pH, U, 0.75, figsize, cmap)
        add_text(ax, text, offset=0.05)

        if savefig:
            plt.savefig(savefig, dpi=300)
        if show:
            plt.show()

        return ax
    '''


def main():
    import matplotlib.pyplot as plt

    material = 'TiO2'
    refs = get_references(
        material,
        '/home/niflheim/steame/utils/oqmd123.db',
        computed_energy=None,
        include_aq=True
    )

    phrange=[0, 14]
    urange=[-3, 3]    
    pbx = Pourbaix(material, refs, conc=1e-24)
    pbx.draw_diagram_axes(urange, phrange)
    



if __name__ == '__main__':
    main()
