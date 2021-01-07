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
    column1 = [fig('bilayer_scatter.png')]
    desc = 'Estimated exfoliation energy [eV/Ang^2]'
    exfoliation_row = [describe_entry('Exfoliation energy', description=desc), f'{result.exfoliation_energy:0.4f} eV/Ang^2']
    column2 = [exfoliation_row]

    panel = {'title': title,
             'columns': [column1, column2],
             'plot_descriptions': [{'function': scatter_energies,
                                    'filenames': ['bilayer_scatter.png']}],
             'sort': 12}

    return [panel]


def scatter_energies(row, fname):
    d = row.data.get('results-asr.bilayersummary.json')['binding_data']
    import matplotlib.pyplot as plt

    energies = []
    lengths = []
    data = np.array(list(d.values()))

    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig(fname)
        

@prepare_result
class Result(ASRResult):
    """Container for summary results."""
    
    binding_data: dict
    exfoliation_energy: float

    key_descriptions = dict(
        binding_data='Key: Bilayer descriptor, value: binding energy [eV / Ang^2] and binding length [Ang]',
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
    p = Path(monolayerfolder)
    for sp in [x for x in p.iterdir() if x.is_dir()]:
        binding_path = f"{sp}/results-asr.bilayer_binding.json"
        if os.path.exists(binding_path):
            data = read_json(binding_path)
            binding_data[str(sp)] = (data["binding_energy"], data['interlayer_distance'])
                
    return Result.fromdata(binding_data=binding_data,
                           exfoliation_energy=max(map(lambda t: t[0], binding_data.values())))
