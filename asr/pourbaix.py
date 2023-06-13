import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pfx
from itertools import product, chain, combinations
from collections import Counter, OrderedDict
from typing import List, Tuple, Dict, Union
from ase.io.jsonio import read_json
from ase.formula import Formula, non_metals
from ase import units
from asr.core import command, option, ASRResult, prepare_result


PREDEF_ENERGIES = {
    'O': -4.57,     # http://dx.doi.org/10.1103/PhysRevB.85.235438
    'H': -3.73,     # http://dx.doi.org/10.1103/PhysRevB.85.235438
    'H+': 0.0,
    'H2O': -2.4583,
    'OH-': -2.54    # https://doi.org/10.1021/jp063552y
}


#---Classes---#

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
        return f'Species({self.name}, μ={self.mu})'

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name


class RedOx:
    '''
    Balances the chemical equation for a given phase
    and pre-calculates the corresponding 
    reaction free energy
    '''
    def __init__(self, reactant, products,
                 T=298.15, conc=1e-6,
                 env='acidic', counter=False):
        self.reactant = reactant.name
        self.products = [p.name for p in products]
        self.species = {reactant.name: -1}
        self.env = env

        const = units.kB * T
        alpha = np.log(10) * const
        reac_charge = reactant.charge
        gibbs0 = -reactant.mu
        total = Counter()

        # Updating element count with reactant
        total.subtract(reactant.count)

        # Updating element, energy and charge count with products
        prod_charge = 0
        for prod in products:
            if prod == reactant:
                N = 1
            else:
                n_elem = []
                for elem, n in prod.count.items():
                    if elem in reactant.elements:
                        n_elem.append(reactant.count[elem] / n)
                N = max(n_elem)
            for elem, n in prod.count.items():
                total[elem] += N * n

            self.species[prod.name] = N
            gibbs0 += N * (prod.mu + prod.aq * const * np.log(conc))
            prod_charge += N * prod.charge

        # Completing count by adding elemental phases, if necessary
        # e.g. GeSe + Se --> GeSe2
        for elem in reactant.elements:
            if total[elem] != 0:
                if elem in self.species:
                    self.species[elem] -= total[elem]
                else:
                    self.species[elem] = -total[elem]
                total.pop(elem)

        # Balancing O atoms with water
        n_H2O = - total['O']
        total['H'] += 2 * n_H2O
        total['O'] += n_H2O

        # Balancing H atoms with H+
        n_H = - total.pop('H')
        if n_H > 0:
            prod_charge += n_H
        else:
            reac_charge -= n_H

        assert(np.allclose(list(total.values()), 0.0, rtol=0.0, atol=1e-6)), \
            f"Reaction for {', '.join(self.products)} incorrectly balanced..."

        # Balancing residual charge with electrons
        n_e = prod_charge - reac_charge
        self.species['e-'] = n_e
        n_OH = 0

        # Adjusting the equation for alkaline environment
        if env == 'alkaline':
            n_H2O = n_H + n_H2O
            n_OH = - n_H
            n_H = 0

        for spc, number in zip(['H+', 'OH-', 'H2O'], [n_H, n_OH, n_H2O]):
            self.species[spc] = number
            gibbs0 += number * PREDEF_ENERGIES[spc]

        # Storing relevant constants for later use in
        # free energy evaluation
        K = gibbs0 + 14.0 * n_OH * alpha
        C = n_H - n_OH

        if counter:
            C += n_e
            K += get_counter_correction(env, n_e, alpha)

        self._vector = [K, -n_e, -alpha * C]

    def equation(self):
        if self.species[self.reactant] == 1:
            return self.reactant
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

    def evaluate(self, U, pH):
        '''Evaluate Nernst equation at given pH and U'''
        K, C1, C2 = self._vector
        energy = K + np.dot((C1, C2), (U, pH))
        return energy

    def __eq__(self, other):
        return all(prod in other.products for prod in self.products)


class Pourbaix:
    def __init__(self, material, refs, T=298.15, conc=1.0e-6, counter=False):
        self.material = material
        self.phases, phase_matrix = get_phases(material, refs, T, conc, counter)
        self._const = phase_matrix[:, 0]
        self._var = phase_matrix[:, 1:]
        self.alpha = units.kB * T * np.log(10)

    def _evaluate(self, U, pH):
        return self._const + np.dot(self._var, [U, pH])

    def _decompose(self, U, pH):
        energies = self._evaluate(U, pH)
        i_min = np.argmin(energies)
        return energies[i_min], i_min

    def decompose(self, U, pH, verbose=True):
        energy, index = self._decompose(U, pH)
        phase = self.phases[index]
        if verbose:
            print(f'Stable phase: \n{phase.equation()}'
                  f'\nEnergy: {energy} eV')
        return energy, phase

    def get_diagrams(self, U, pH):
        pour = np.zeros((len(U), len(pH)))
        meta = pour.copy()

        prodstring = lambda phase: ",".join(phase.products)

        for i, u in enumerate(U):
            for j, p in enumerate(pH):
                meta[i, j], pour[i, j] = self._decompose(u, p)

        text = []
        for num in np.unique(pour):
            phase = self.phases[int(num)]
            pstr = prodstring(phase)
            where = (pour == num)
            x = np.dot(where.sum(1), U) / where.sum()
            y = np.dot(where.sum(0), pH) / where.sum()
            text.append((x, y, phase.products))
        return pour, meta, text

    def get_pH_slice(self, pH_value, U, show=True, savefig=None):
        '''Obtain a energy-vs-potential slice of the Pourbaix diagram at a given pH'''

        energies = np.zeros((len(U), len(self.phases)))
        for i, u in enumerate(U):
            energies[i] = self._evaluate(u, pH_value)

        ax = draw_slice_axes(energies, U, self.phases, show, savefig, ptype='pH')
        ax.set_title(f'pH = {pH_value}')
        ax.set_xlabel('U (V)')

        if show:
            plt.show()
        if savefig is not None:
            plt.savefig(savefig)

        return energies

    def get_U_slice(self, U_value, pH, show=True, savefig=None):
        '''Obtain a energy-vs-pH slice of the Pourbaix diagram at a given potential'''

        energies = np.zeros((len(pH), len(self.phases)))
        for i, p in enumerate(pH):
            energies[i] = self._evaluate(U_value, p)

        ax = draw_slice_axes(energies, pH, self.phases, show, savefig, ptype='U')
        ax.set_title(f'U = {U_value}')
        ax.set_xlabel('pH')

        if show:
            plt.show()
        if savefig is not None:
            plt.savefig(savefig)

        return energies

    def plot(self, U, pH, 
             cmap="RdYlGn",
             cap=1.0,
             normalize=True,
             savefig='pourbaix.png',
             figsize=[12, 6],
             show=True):

        pour, meta, text = self.get_diagrams(U, pH)
        if normalize:
            meta /= self.material.natoms
        if cap:
            np.clip(meta, a_min=-cap, a_max=0.0, out=meta)

        ax = draw_diagram_axes(pour, meta, text, pH, U, 0.75, figsize, cmap)
        add_text(ax, text, offset=0.05)

        if savefig:
            plt.savefig(savefig, dpi=300)
        if show:
            plt.show()

        return ax


def get_counter_correction(env, n_e, alpha):
    std_potentials = {
        'acidic': {
            1: 0.0,       # 2H+ + 2e- --> H2
           -1: 1.229,     # 2H2O --> O2 + 4H+ + 4e-
            0: 0.0
        },
        'alkaline': {
            1: -0.8277,   # 2H2O + 2e- --> H2 + 2OH-
           -1: 0.401,     # 4OH- --> O2 + 2H2O + 4e-
            0: 0.0
        }
    } 
    E0 = std_potentials[env][np.sign(n_e)]
    if env == 'alkaline':
        return -n_e * (E0 + 14.0 * alpha)

    return -n_e * E0


#---Plotting-functions---#

def edge_detection(array):
    edges = {}
    for i in range(array.shape[0] - 1):
        for j in range(array.shape[1] - 1):
            xpair = (array[i, j], array[i+1, j])
            ypair = (array[i, j], array[i, j+1])
            for pair in [xpair, ypair]:
                if np.ptp(pair) != 0:
                    if pair in edges.keys():
                        edges[pair].append([i+1, j])
                    else:
                        edges.update({
                            pair: [[i+1, j]]
                        })
    for key, value in edges.items():
        edges.update({key: np.asarray(value)})
    return edges


def add_edges(axes, diagram, pH, U, **kwargs):

    def get_linear_fit(X, Y, indexes):
        Xvals = [X[i] for i in indexes[:, 1]]
        Yvals = [Y[i] for i in indexes[:, 0]]
        if np.ptp(Xvals) == 0.0:
            return Xvals, Yvals
        fit = np.polyfit(Xvals, Yvals, deg=1)
        Yfit = [x * fit[0] + fit[1] for x in Xvals]
        return Xvals, Yfit

    # add edges to diagram
    edges = edge_detection(diagram)
    for colors, indexes in edges.items():
        x, y = get_linear_fit(pH, U, indexes)
        axes.plot(
            x, y,
            ls='-',
            marker=None,
            zorder=1,
            **kwargs
        )
    return 0


def add_redox_lines(axes, pH, color='k'):
    # Add water redox potentials
    slope = -59.2e-3
    axes.plot(pH, slope*pH, c=color, ls='--', zorder=2)
    axes.plot(pH, slope*pH + 1.229, c=color, ls='--', zorder=2)
    return 0


def add_numbers(ax, text):
    for i, (x, y, prod) in enumerate(text):
        txt = ax.text(
            y, x, f'{i}', 
            fontsize=20,
            horizontalalignment='center'
        )
        txt.set_path_effects([pfx.withStroke(linewidth=3.0, foreground='w')])
    return
        
    
def add_text(ax, text, offset=0):
    import textwrap

    textlines = []
    for i, (x, y, prod) in enumerate(text):
        # Adding text on the diagram
        label = ' + '.join(i for i in prod)
        label = re.sub(r'(\S)([+-]+)', r'\1$^{\2}$', label)
        label = re.sub(r'(\d+)', r'$_{\1}$', label)
        for symbol in ['+', '-']:
            count = label.count('+')
            if count > 1:
                label = label.replace(count*symbol, f'{count}{symbol}')
            if count == 1:
                label = label.replace(count*symbol, symbol)
        textlines.append(
                textwrap.fill(f'({i})  {label}', 
                              width=40,
                              subsequent_indent='      ')
        )

    text = "\n".join(textlines)
    plt.gcf().text(
            0.74 + offset, 0.5,
            text,
            fontsize=18,
            va='center',
            ha='left')
    return 0


def draw_diagram_axes(phases, diagram, text, pH, U, right, figsize=None, cmap='RdYlGn'):
    ax = plt.figure(figsize=figsize).add_subplot(111)
    plt.subplots_adjust(
        left=0.1, right=right,
        top=0.97, bottom=0.14
    )
    plot_opts = {
            'cmap': cmap,
            'extent': [
                min(pH), max(pH),
                min(U), max(U)
            ],
            'origin': 'lower',
            'aspect': 'auto',
            'vmax': 0.0
    }

    colorplot = ax.imshow(diagram, **plot_opts)
    cbar = plt.gcf().colorbar(
           colorplot,
           ax=ax,
           pad=0.02
    )

    ax.set_xlabel('pH', fontsize=20)
    ax.set_ylabel('Applied potential [V]', fontsize=20)
    ax.set_xlim(min(pH), max(pH))
    ax.set_ylim(min(U), max(U))
    ax.set_xticks(np.arange(min(pH), max(pH) + 1, 2))
    ax.set_yticks(np.arange(min(U), max(U) + 1, 1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ticks = np.linspace(diagram.min(), 0, num=6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['{:.1f}'.format(abs(tick)) for tick in ticks])
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(r'$\mathrm{\Delta E}$ / atom [eV]', fontsize=20)

    add_numbers(ax, text)
    add_edges(ax, phases, pH, U, ms=0.5, color='k')
    add_redox_lines(ax, pH, 'w')

    return ax


def draw_slice_axes(energies, X, phases, show, savefig, ptype):
    from matplotlib.patches import Rectangle
    ax = plt.figure().add_subplot(111)
    ax.add_patch(Rectangle(
        (X.min(), 0),
        np.ptp(X),
        energies.max(),
        color='#fbcdc8'
    ))

    ax.plot(X, energies[:, 0], 'k--', zorder=len(phases)+1)
    for i in range(1, energies.shape[1]):
        phase = phases[i]
        label = ', '.join(phase.products)
        if ptype == 'pH':
            label += f", {phase.species['e-']:.1f} e-"
        else:
            label += f", {phase.species['H+']:.1f} H+"
        ax.plot(X, energies[:, i], label=label)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(energies.min(), energies.max())
    ax.set_ylabel(r'$\Delta G$ (eV)')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=10)
    plt.subplots_adjust(
        left=0.15, right=0.75,
        top=0.9, bottom=0.14
    )
    return ax


def generate_figures(row, filename):
    '''Plot diagram for webpanel'''

    data = row.data.get('results-asr.pourbaix.json')
    phases = data['phases']
    diagram = data['diagram']
    text = data['names']
    count = data['count']
    points = data['evaluation_points']
    energies = data['material_energy']
    pH = data['pH']
    U = data['U']
    natoms = sum(count.values())

    diagram /= natoms
    np.clip(diagram, a_min=-1, a_max=0.0, out=diagram)
    ax = draw_diagram_axes(phases, diagram, text, pH, U, 1.0)

    for point, energy in zip(points, energies):
        ax.scatter(
            *point,
            c='w',
            s=80,
            marker='o',
            edgecolors='k',
            linewidths=1.5,
            zorder=3,
            vmax=0.0,
            vmin=-1,
        )

    plt.savefig(filename)
    
    return ax


#---Diagram-generation-functions---#

def get_references(material, db_name, computed_energy=None, include_aq=True):
    from ase.db import connect
    from ase.phasediagram import solvated

    reference_energies = {}
    refs = {}

    db = connect(db_name)
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
                reference_energies[name] = energy_elem

            OH_content = ref.get_fractional_composition(['O', 'H'])
            if OH_content > 0.85 or 1e-4 < OH_content < 0.095:
                continue

            ref.set_chemical_potential(energy, reference_energies)
            refs[name] = ref

    mat = refs.get(material.name, None)
    if not mat:
        if computed_energy is None:
            raise ValueError(\
                f'Your material has not been found in {db_name}. '
                f'Please provide an energy for it!'
            )
        material.set_chemical_potential(computed_energy, reference_energies)
    else:
        material.set_chemical_potential(mat.mu, None)

    refs[material.name] = material

    if include_aq:
        solv_refs = solvated(material.name)
        for name, energy in solv_refs:
            ref = Species(name)
            ref.set_chemical_potential(energy, None)
            refs[name] = ref

    return refs


def get_phases(material, refs, T, conc, counter):
    '''Find all possible reaction pathways for a given material'''

    contains = lambda mat, elements: \
        [True if elem in mat.elements else False for elem in elements]

    array = [[] for i in range(len(material.elements))]
    others = refs.copy()
    others.pop(material.name)
    for other in others.values():
        contained = contains(other, material.elements)
        for w in np.argwhere(contained).flatten():
            array[w].append(other)

    # Initializing phases with material itself
    no_reaction = RedOx(material, (material,), T, conc, counter=counter)
    phases = [no_reaction]
    phase_matrix = [no_reaction._vector]

    for combo in product(*array):
        # Sometimes product can give combinations 
        # with equal elements. We don't want that
        if len(np.unique(combo)) < len(combo):
            continue

        #TODO Add alkaline environment once chemical potentials are figured out
        #for env in ['acidic', 'alkaline']:
        for env in ['acidic']:
            phase = RedOx(material, combo, T, conc, env, counter)
            # Avoid duplicates
            if env == 'alkaline' and phase.species['OH-'] == 0:
                continue
            phases.append(phase)
            phase_matrix.append(phase._vector)

    return phases, np.array(phase_matrix)


def get_pourbaix_diagram(
        material,
        energy=None,
        T=298.15, conc=1e-6,
        counter=False,
        database='/home/niflheim/steame/utils/oqmd123.db'):

    if isinstance(material, str):
        material = Species(material)

    refs = get_references(material, database, computed_energy=energy)
    refs.pop('H+(aq)')
    refs.pop('H2O(aq)')

    pourbaix = Pourbaix(material, refs, T, conc, counter)

    return pourbaix, refs


#---ASR-stuff---#

def make_results(pourbaix, refs, points, U, pH):
    phases, diagram, text = pourbaix.get_diagrams(U, pH)
    natoms = pourbaix.material.natoms
    results = {
        'phases': phases,
        'diagram': diagram,
        'names': text,
        'count': pourbaix.material.count,
        'pH': pH,
        'U': U,
        'evaluation_points': [],
        'material_energy': [],
        'references': []
    }

    for point in points:
        energy, phase = pourbaix.decompose(
            point[1], point[0], verbose=False
        )
        results['evaluation_points'].append(point)
        results['material_energy'].append(-energy/natoms)

    for ref in refs.values():
        results['references'].append([ref.name, ref.mu])

    return results


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (
        table, matrixtable, fig,
        describe_entry, WebPanel,
        make_panel_description
    )
    description = """\
    The thermodynamic stability of the material in aqueous solution
    as a function of pH and the externally applied potential.
    
    The colormap displays the energy difference ΔE 
    (normalized wrt. the number of atoms and unit formulas)
    between the material and the most stable phase in that region
    of the diagram. For reference, the plot also displays the H+/H2
    and O2/H2O reduction potentials as a function of pH
    (white dashed lines). The phases and their IDs are listed
    in the table on the right.
    
    The bottom table shows the ΔE values at selected 
    pH and applied potentials. The corresponding (pH, U) points are
    highlighted in the diagram.
    """
    panel_description = make_panel_description(description)

    points = result['evaluation_points'].copy()
    energies = result['material_energy'].copy()
    text = result['names'].copy()

    rows1 = []
    for pt, en in zip(points, energies):
        rows1.append([f'{pt[0]:.0f}', f'{pt[1]:.2f}', f'{en:.2f}'])

    table1 = {
        'type': 'table',
        'header': ['pH', 'U [V]', 'ΔE/atom [eV]'],
        'rows': rows1
    }

    rows2 = []
    for i, (x, y, prod) in enumerate(text):
        label = ' + '.join(i for i in prod)
        label = re.sub(r'(\S)([+-]+)', r'\1<sup>\2</sup>', label)
        label = re.sub(r'(\d+)', r'<sub>\1</sub>', label)
        label = label.replace('--', '2-').replace('++', '2+')
        rows2.append([i, label])

    table2 = {
        'type': 'table',
        'header': ['ID', 'Phase'],
        'rows': rows2
    }

    panel = WebPanel(
        title=describe_entry(
            'Pourbaix diagram',
             description=panel_description),
        columns=[[fig('pourbaix12.png'), table1], [table2]],
        plot_descriptions=[{
            'function': generate_figures,
            'filenames':['pourbaix12.png']
        }]
    )

    return [panel] 


@prepare_result
class Result(ASRResult):
    '''Container for Pourbaix diagram results'''
    phases: np.array
    diagram: np.array
    pH: np.array
    U: np.array
    count: dict
    names: List[List]
    evaluation_points: List[List[float]]
    material_energy: List[float]
    references: List[List]

    key_descriptions = {
            'phases': 'Pourbaix diagram showing the most stable phases',
            'diagram': (
                'Pourbaix diagram with the energies ',
                'relative to the reference material'),
            'names': 'Coordinates and text of the phase labels',
            'count': 'Formula count of the material',
            'pH': 'pH axis',
            'U': 'Potential axis',
            'evaluation_points': (
                'List of (pH, U) points in the diagram '
                'at which the material stability was evaluated'),
            'material_energy': (
                'Energy (per atom per unit formula) of the material, '
                'relative to the most stable phase, at each evaluation point'),
            'references': (
                'Formula and chemical potential of the solid and aqueous references '
                'used for generating the diagram')
    }

    formats = {"ase_webpanel": webpanel}


@command(module='asr.pourbaix',
         requires=['gs.gpw'],
         returns=Result)
@option('--gpw', type=str, help='GPAW calculated density')
@option('--temp', type=float, help='Temperature in K')
@option('--conc', type=float, help='Concentration of aqueous species')
@option('--phrange', type=list, help='Lower and upper pH limits for the Pourbaix diagram')
@option('--urange', type=list, help='Lower and upper potential limits for the Pourbaix diagram')
@option('--npoints', type=int, help='Number of grid points along the pH axis. The U axis is subdivided accordingly')
@option('--counter', type=bool, help='Apply counter-electrode corrections')
@option('--database', type=str, help='Database containing the solid references')
@option('--eval-points', type=list, help='List of (pH, U) points at which the material stability is evaluated')
def main(
        gpw='gs.gpw',
        temp=298.15,
        conc=1e-6,
        phrange=[-3, 3],
        urange=[-1, 15],
        npoints=600,
        counter=False,
        database='/home/niflheim2/cmr/databases/referencedatabases/oqmd123.db',
        eval_points: [List[Tuple]]=[
           (0.0, 0.0),
           (0.0, 1.229),
           (7.0, -0.413),
           (7.0, 0.817),
           (14.0, -0.828),
           (14.0, 0.401)
        ]) -> Result:
    from gpaw import GPAW
    from ase.db import connect

    calc = GPAW(gpw)
    material = Species(calc.atoms.get_chemical_formula())
    computed_energy = calc.get_potential_energy()

    pourbaix, refs = get_pourbaix_diagram(
        material, computed_energy,
        temp, conc, counter,
        database
    )

    ratio = np.ptp(urange) / np.ptp(phrange)
    pH = np.linspace(*phrange, num=npoints)
    U = np.linspace(*urange, num=int(npoints * ratio))

    pourbaix.plot(pH, U)
    results = make_results(pourbaix, refs, eval_points, U, pH)

    return results


if __name__ == '__main__':
    main.cli()
