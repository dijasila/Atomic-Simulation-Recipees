from ase.io import read, write
from asr.core import command, option
from gpaw import GPAW, restart
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc


@command(module='asr.analyze_state',
         requires=['gs.gpw', 'structure.json',
                   '../../defects.pristine_sc/gs.gpw'],
         resources='24:2h')
@option('--state', help='Specify the specific state (band number) that you '
        'want to consider. Note, that this argument is not used when the '
        'gap state flag is active.', type=int)
@option('--get-gapstates/--dont-get-gapstates', help='Should all of the gap'
        ' states be saved and analyzed? Note, that if gap states are analysed'
        ' the --state option will be neglected.', is_flag=True)
@option('--analyze/--dont-analyze', help='Not only create cube files of '
        'specific states, but also analyze them.', is_flag=True)
def main(state: int = 0,
         get_gapstates: bool = False,
         analyze: bool = False):
    """Write out wavefunction and analyze it.

    This recipe reads in an existing gs.gpw file and writes out wavefunctions
    of different states (either the one of a specific given bandindex or of
    all the defect states in the gap). Furthermore, it will feature some post
    analysis on those states.

    Test.
    """
    atoms = read('structure.json')
    print('INFO: run fixdensity calculation')
    calc = GPAW('gs.gpw', txt='analyze_states.txt')
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})
    if get_gapstates:
            print('INFO: evaluate gapstates ...')
            states = return_gapstates_fix(calc, spin=0)
    elif not get_gapstates:
        states = [state]

    print('INFO: write wavefunctions of gapstates ...')
    for band in states:
        wf = calc.get_pseudo_wave_function(band=band, spin=0)
        fname = 'wf.{0}_{1}.cube'.format(band, 0)
        write(fname, atoms, data=wf)
        if calc.get_number_of_spins() == 2:
            wf = calc.get_pseudo_wave_function(band=band, spin=1)
            fname = 'wf.{0}_{1}.cube'.format(band, 1)
            write(fname, atoms, data=wf)

    print('INFO: Calculating dipole matrix elements among gap states.')
    d_svnm = dipole_matrix_elements_from_calc(calc, n1=states[0], n2=states[-1]+1)

    if analyze:
        # To be implemented
        print('INFO: analyze chosen states.')

    results = {'states': states,
               'dipole': d_svnm}

    return results


def plot_gapstates(row, fname):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    evbm, ecbm, gap = get_band_edge()

    # Draw bands edge
    draw_band_edge(evbm, 'vbm', 'C0', offset=gap/5, ax=ax)
    draw_band_edge(ecbm, 'cbm', 'C1', offset=gap/5, ax=ax)
    # Loop over eigenvalues to draw the level
    calc = GPAW('gs.gpw')
    nband = calc.get_number_of_bands()
    ef = calc.get_fermi_level()

    for s in range(calc.get_number_of_spins()):
        for n in range(nband):
            ene = calc.get_eigenvalues(spin=s, kpt=0)[n]
            occ = calc.get_occupation_numbers(spin=s, kpt=0)[n]
            enenew = calc.get_eigenvalues(spin=s, kpt=0)[n+1]
            print(n, ene, occ)
            lev = Level(ene, ax=ax)
            if (ene >= evbm + 0.05 and ene <= ecbm - 0.05):
                # check degeneracy
                if abs(enenew - ene) <= 0.01:
                    lev.draw(spin=s, deg=2)
                elif abs(eneold - ene) <= 0.01:
                    continue
                else:
                    lev.draw(spin=s, deg=1)
                # add arrow if occupied
                if ene <= ef:
                    lev.add_occupation(length=gap/10)
             if ene >= ecbm:
                break
             eneold = ene

    # plotting
    ax.plot([0,1],[ef]*2, '--k')
    ax.set_xlim(0,1)
    ax.set_ylim(evbm-gap/5,ecbm+gap/5)
    ax.set_xticks([])
    ax.set_ylabel('Energy (eV)', size=15)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def get_band_edge():
    calc = GPAW('../../defects.pristine_sc/gs.gpw')
    gap, p1, p2 = bandgap(calc)
    evbm = calc.get_eigenvalues(spin=p1[0], kpt=p1[1])[p1[2]]
    ecbm = calc.get_eigenvalues(spin=p2[0], kpt=p2[1])[p2[2]]
    return evbm, ecbm, gap

def draw_band_edge(energy, edge, color, offset=2, ax=None):
    if edge == 'vbm':
        eoffset = energy - offset
        elabel = energy - offset/2
    elif edge == 'cbm':
        eoffset = energy + offset
        elabel = energy + offset/2

    ax.plot([0,1],[energy]*2, color=color, lw=2,zorder=1)
    ax.fill_between([0,1],[energy]*2,[eoffset]*2, color=color, alpha=0.7)
    ax.text(0.5, elabel, edge.upper(), color='w', fontsize=18, ha='center', va='center')

class Level:
    "Class to draw a single defect state level in the gap, with an
     arrow if occupied. The direction of the arrow depends on the
     spin channel"

    def __init__(self, energy, size=0.05, ax=None):
        self.size = size
        self.energy = energy
        self.ax = ax

    def draw(self, spin, deg):
        """Draw the defect state according to its 
           spin  and degeneracy"""

        relpos = [[1/4,1/8],[3/4,5/8]][spin][deg-1]
        pos = [relpos - self.size, relpos + self.size]
        self.relpos = relpos
        self.spin = spin
        self.deg = deg

        if deg == 1:
            self.ax.plot(pos, [self.energy] * 2, '-k')

        if deg == 2:
            newpos = [p + 1/4 for p in pos]
            self.ax.plot(pos, [self.energy] * 2, '-k')
            self.ax.plot(newpos, [self.energy] * 2, '-k')

    def add_occupation(self, length):
        "Draw an arrow if the defect state is occupied"

        updown = [1,-1][self.spin]
        self.ax.arrow(self.relpos, self.energy - updown*length/2, 0, updown*length, head_width=0.01, head_length=length/5, fc='k', ec='k')
        if self.deg == 2:
            self.ax.arrow(self.relpos + 1/4, self.energy - updown*length/2, 0, updown*length, head_width=0.01, head_length=length/5, fc='k', ec='k')

def return_gapstates(calc_def, spin=0):
    """Evaluates which states are inside the gap and returns the band indices
    of those states for a given spin channel.
    """
    from asr.core import read_json

    _, calc_pris = restart('../../defects.pristine_sc/gs.gpw', txt=None)
    results_pris = read_json('../../defects.pristine_sc/results-asr.gs.json')
    results_def = read_json('results-asr.gs.json')
    vbm = results_pris['vbm'] - results_pris['evac']
    cbm = results_pris['cbm'] - results_pris['evac']

    es_def = calc_def.get_eigenvalues() - results_def['evac']
    es_pris = calc_pris.get_eigenvalues() - results_pris['evac']

    diff = es_pris[0] - es_def[0]
    states_def = es_def + diff

    statelist = []
    [statelist.append(i) for i, state in enumerate(states_def) if (
        state < cbm and state > vbm)]

    return statelist


def return_gapstates_fix(calc_def, spin=0):
    """HOTFIX until spin-orbit works with ASR!"""

    _, calc_pris = restart('../../defects.pristine_sc/gs.gpw', txt=None)
    evac_pris = calc_pris.get_electrostatic_potential()[0,0,0]
    evac_def = calc_def.get_electrostatic_potential()[0,0,0]

    vbm, cbm = calc_pris.get_homo_lumo() - evac_pris

    es_def = calc_def.get_eigenvalues() - evac_def
    es_pris = calc_pris.get_eigenvalues() - evac_pris

    diff = es_pris[0] - es_def[0]
    states_def = es_def + diff

    statelist = []
    [statelist.append(i) for i, state in enumerate(states_def) if (
        state < cbm and state > vbm)]

    return statelist


if __name__ == '__main__':
    main.cli()
