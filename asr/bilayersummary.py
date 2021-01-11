import typing
from asr.core import command, ASRResult, prepare_result, option, read_json
from asr.database.browser import fig, make_panel_description, describe_entry
from pathlib import Path
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
    desc = 'The exfoliation energy is estimated by taking binding energy of the most tightly bound bilayer stacking.'
    exfoliation_row = [describe_entry('Exfoliation energy', description=desc), f'{result.exfoliation_energy:0.4f} eV/Ang<sup>2</sup>']
    column2 = [{'type': 'table',
                'header': ['Property', 'Value'],
                'rows': [exfoliation_row]}]

    panel = {'title': title,
             'columns': [column1, column2],
             'plot_descriptions': [{'function': scatter_energies,
                                    'filenames': ['bilayer_scatter.png']},
                                   {'function': plot_gaps,
                                    'filenames': ['bl_gaps.png']}],
             'sort': 12}

    return [panel]


def scatter_energies(row, fname):
    d = row.data.get('results-asr.bilayersummary.json')['binding_data']
    import matplotlib.pyplot as plt
    data = np.array(list(d.values()))

    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel(r"Energy [eV/Ang$^2$]")
    plt.ylabel(r"Interlayer distance [Ang]")
    plt.savefig(fname)


def make_rectangle(ax, i, g):
    from matplotlib.patches import Rectangle
    x = i
    w = 0.2
    h = g * 2
    
    rect = Rectangle((i - w/2, -h/2), width=w, height=h, color="lightgray", fill=True)
    return rect


def plot_gaps(row, fname):
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


@prepare_result
class Result(ASRResult):
    """Container for summary results."""
    
    binding_data: dict
    gaps: dict
    exfoliation_energy: float

    key_descriptions = dict(
        binding_data='Key: Bilayer descriptor, value: binding energy [eV / Ang^2] and binding length [Ang]',
        gaps='Key: Bilayer descriptor, value: gap [eV]',
        exfoliation_energy='Estimated exfoliation energy [eV]')

    formats = {'ase_webpanel': webpanel}


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
    p = Path(monolayerfolder)
    for sp in [x for x in p.iterdir() if x.is_dir()]:
        # Get binding data
        binding_path = f"{sp}/results-asr.bilayer_binding.json"
        if os.path.exists(binding_path):
            data = read_json(binding_path)
            binding_data[str(sp)] = (data["binding_energy"], data['interlayer_distance'])
                
        # Get gap data
        gs_path = f"{sp}/results-asr.gs.json"
        if os.path.exists(gs_path):
            data = read_json(gs_path)
            gaps[str(sp)] = data["gap"]


    return Result.fromdata(binding_data=binding_data,
                           gaps=gaps,
                           exfoliation_energy=max(map(lambda t: t[0], binding_data.values())))
