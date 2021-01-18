import typing
from asr.core import command, ASRResult, prepare_result, option, read_json
from asr.database.browser import fig, make_panel_description, describe_entry
from asr.database import material_fingerprint as mfp
from asr.structureinfo import get_reduced_formula
from ase.io import read
from pathlib import Path
from asr.bilayerdescriptor import get_descriptor
import os
import numpy as np


panel_description = make_panel_description(
    """Summarizes the result of bilayer calculations.
Shows a binding energy vs. binding length scatter plot
and the estimated monolayer exfoliation energy.""")

def webpanel(result, row, key_descriptions):
    title = describe_entry('Bilayer summary',
                           panel_description)
    column1 = [fig('bilayer_scatter.png'),
               fig('bl_gaps.png')]
    table = make_summary_table(result)
    column2 = [{'type': 'table',
                'header': ['Property', 'Value'],
                'rows': table}]

    panel = {'title': title,
             'columns': [column1, column2],
             'plot_descriptions': [{'function': scatter_energies,
                                    'filenames': ['bilayer_scatter.png']},
                                   {'function': plot_gaps,
                                    'filenames': ['bl_gaps.png']}],
             'sort': 12}

    return [panel]


def make_row_title(desc, result):
    """Make a row title with descriptor and link."""
    link = result['links'].get(desc, '')
    if link == '':
        return desc
    else:
        return f'<a href="{link}">{desc}</a>'
    


def make_summary_table(result):
    desc = 'The exfoliation energy is estimated by taking binding energy of the most tightly bound bilayer stacking.'
    energy = result.exfoliation_energy
    exfoliation_val = f'{energy:0.4f} eV/Ang<sup>2</sup>' if type(energy) != str else energy
    exfoliation_row = [describe_entry('Exfoliation energy', description=desc), exfoliation_val]
    
    items = sorted([(desc, val[0]) for desc, val in result.binding_data.items()], key=lambda t:-t[1])
    binding_rows = [[make_row_title(desc, result), f'{val:0.5f} eV/Ang<sup>2</sup>'] for desc, val in items]

    return [exfoliation_row] + binding_rows

def scatter_energies(row, fname):
    d = row.data.get('results-asr.bilayersummary.json')['binding_data']
    ns = row.data.get('results-asr.bilayersummary.json')['numberings']
    import matplotlib.pyplot as plt
    data = np.array(list(d.values()))

    plt.scatter(data[:, 1], data[:, 0])
    for desc, dat in d.items():
        plt.gca().annotate(ns[desc], (dat[1], dat[0] + 0.75e-4), va="bottom", ha="center",
                           fontsize=12)
    (y1, y2) = plt.ylim()
    plt.ylim((y1, y2 + 3e-4))

    maxe = np.max(data[:, 0])
    cutoff = maxe - 2e-3
    plt.gca().axhline(cutoff, label='Stability cutoff',
                      color="black", linestyle="dashed", linewidth=2)
    if y2 > 0.15:
        cutoff2 = 0.15
        plt.gca().axhline(cutoff2, label='Exfoliability cutoff')
    

    plt.xlabel(r'Interlayer distance [Ang]')
    plt.ylabel(r'Energy [eV/Ang$^2$]')
    plt.legend()
    plt.savefig(fname)


def make_rectangle(ax, i, g):
    from matplotlib.patches import Rectangle
    x = i
    w = 0.2
    h = g * 2
    
    rect = Rectangle((i - w/2, -h/2), width=w, height=h, color="lightgray", fill=True)
    return rect


def plot_bargaps(row, fname):
    d = row.data.get('results-asr.bilayersummary.json')['gaps']
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    n = 0
    for i, (desc, g) in enumerate(d.items()):
        n += 1
        ax.add_patch(make_rectangle(ax, i+1, g))
        
    ax.set_xlim((0, n+1))
    
    Y = max(d.values()) * 1.2
    ax.set_ylim((-Y, Y))
    ax.set_xticks(list(range(1, n+1)))
    ax.set_xlabel("BL stackings")
    ax.set_ylabel("Gap [eV]")
    plt.savefig(fname)


def plot_gaps(row, fname):
    import matplotlib.pyplot as plt
    d = row.data.get('results-asr.bilayersummary.json')

    fig, ax = plt.subplots()
    numbers = d['numberings']
    data = []
    for desc, numb in d['numberings'].items():
        if desc in d['gaps']:
            data.append((numb, d['gaps'][desc]))
    data = np.array(data)
    data = np.sort(data, axis=0)

    ax.plot(data[:, 0], data[:, 1], marker='x', label='Bilayer gaps')
    ax.axhline(d['monolayer_gap'], label='Monolayer gap', linewidth=2, color='black',
               linestyle='dashed')
    # ax.set_xticks(range(1, len(gaps) + 1))
    ax.set_xlabel("BL stackings")
    ax.set_ylabel("Gap [eV]")
    ax.legend()
    plt.savefig(fname)


@prepare_result
class Result(ASRResult):
    """Container for summary results."""
    
    binding_data: dict
    gaps: dict
    links: dict
    numberings: dict
    exfoliation_energy: float
    monolayer_gap: float

    key_descriptions = dict(
        binding_data='Key: Bilayer descriptor, value: binding energy [eV / Ang^2] and binding length [Ang]',
        gaps='Key: Bilayer descriptor, value: gap [eV]',
        links='Key: Bilayer descriptor, value: info to make link',
        numberings='Key: Bilayer descriptor, value: Numbering based on stability',
        exfoliation_energy='Estimated exfoliation energy [eV]',
        monolayer_gap='Monolayer gap [eV]')

    formats = {'ase_webpanel': webpanel}


def make_numbering(binding_data, monolayerpath):
    p = Path(monolayerpath)
    numberings = {}
    numbers = list(enumerate(sorted([(desc, e[0]) for (desc, e) in binding_data.items()],
                                    key=lambda t:-t[1])))
    if len(numbers) == 0:
        return numberings

    maxn = max(numbers, key=lambda t:t[0])
    numbers = {desc: i+1 for i, (desc, _) in numbers}
        
    notnumbered = []
    for sp in [x for x in p.iterdir() if x.is_dir()]:
        desc = get_descriptor(str(sp))
        if desc in binding_data:
            numberings[desc] = numbers[desc]
        else:
            notnumbered.append(desc)
    for desc in notnumbered:
        maxn = maxn + 1
        numberings[desc] = maxn
    return numberings



@command(module='asr.bilayersummary',
         returns=Result)
@option('--monolayerfolder', type=str)
def main(monolayerfolder: str = "./") -> Result:
    """Summarize bilayer calculations.

    Looks through subfolders of monolayer_folder and extracts
    binding energies for each subfolder that has the required
    files.
    """

    binding_data = {}
    gaps = {}
    links = {}
    monolayer_gap = None

    p = Path(monolayerfolder)
    for sp in [x for x in p.iterdir() if x.is_dir()]:
        desc = get_descriptor(str(sp))

        # Get binding data
        binding_path = f"{sp}/results-asr.bilayer_binding.json"
        if os.path.exists(binding_path):
            data = read_json(binding_path)
            binding_data[desc] = (data["binding_energy"], data['interlayer_distance'])
                
        # Get gap data
        gs_path = f"{sp}/results-asr.gs.json"
        if os.path.exists(gs_path):
            data = read_json(gs_path)
            gaps[desc] = data["gap"]

        # Get link info
        struct_path = f'{sp}/structure.json'
        if os.path.exists(struct_path):
            atoms = read(struct_path)
            hsh = mfp.get_hash_of_atoms(atoms)
            uid = mfp.get_uid_of_atoms(atoms, hsh)
            links[desc] = uid

    numberings = make_numbering(binding_data, monolayerfolder)
            
    # Get monolayer gap
    # Go to C2DB-ASR tree to find data
    ml_atoms = read(f'{p}/structure.json')
    formula = str(ml_atoms.symbols.formula)
    stoich = get_reduced_formula(formula, stoichiometry=True)
    reduced = get_reduced_formula(formula, stoichiometry=False)
    full = p.resolve().name
    c2db_path = f'/home/niflheim2/cmr/C2DB-ASR/tree/{stoich}/{reduced}/{full}/results-asr.gs.json'

    if os.path.exists(c2db_path):
        ml_data = read_json(c2db_path)
        monolayer_gap = ml_data['gap']

    return Result.fromdata(binding_data=binding_data,
                           gaps=gaps,
                           links=links,
                           numberings=numberings,
                           exfoliation_energy=max(map(lambda t: t[0], binding_data.values()),
                                                  default='no data'),
                           monolayer_gap=monolayer_gap)



















