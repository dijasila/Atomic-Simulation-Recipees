from gpaw.typing import Array1D, Array2D, Array3D
from ase.units import Bohr, Ha
from typing import Generator, List, Tuple
import numpy as np
import ase.db
import matplotlib.pyplot as plt
from ase.visualize import view
import glob
from pathlib import Path
import os
from ase.io import read, write
import json
from pathlib import Path
from asr.core import chdir
from .mass import extract_stuff_from_gpw_file, fit
from math import pi
from asr.core import write_json, read_json
from asr.database.browser import fig, make_panel_description, describe_entry



panel_description = make_panel_description(
   """cbm effektive mass plot""")

def webpanel(result, row, key_descriptions):
    from asr.database.browser import WebPanel

    data= row.data.get('results-asr.cbm_mass.json')

    extrematable = []
    for xfit, yfit, indices,x, eigs, k, energy, mass, spin  in data['extrema']:
        extrematable.append([f'{k:.3f}', f'{energy:.3f}', f'{mass:.3f}',f'{spin[0]:.1f},{spin[1]:.1f},{spin[2]:.1f}'])



    
    panel = WebPanel(describe_entry(f'Effective masses (cbm)', panel_description),
             columns=[[fig('cbm_mass.png')], []],
             plot_descriptions=[{'function': plot_cbm,
                                    'filenames': ['cbm_mass.png']}],
             sort=7)


    table = {'type': 'table',
             'header': ['Kpoint', 'Energy','Mass','Spin'],
             'rows': extrematable}



    panel2 = WebPanel(title= 'Effective masses (cbm)',
              columns= [[table]],
              sort=7)


    return [panel,panel2]

from asr.core import command, option, ASRResult
@command('asr.cbm_mass')
@option('--name', type=str)
def main(name: str = 'dos.gpw'):
    name=Path(name) 
    (kpoints, length, fermi_level,
     eigenvalues,fingerprints,
     spinprojections) = extract_stuff_from_gpw_file(name)  
 
    extrema = fit(kpoints * 2 * pi / length,
                  fermi_level,
                  eigenvalues,
                  fingerprints, 
                  spinprojections,
                  kind='cbm') 
 
    
   
    results_dict={'extrema': extrema}
    print(results_dict)
    return Result(data=results_dict)


class Result(ASRResult):
    extrema:dict
    key_descriptions = {"cbm_mass" : "cbm effective mass"} 
    formats = {"ase_webpanel": webpanel}


def plot_cbm(row, fname):
    import json
    import numpy as np
    import matplotlib.pyplot as plt



    
    data= row.data.get('results-asr.cbm_mass.json')

    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()
    
    for  xfit, yfit, indices,x, eigs, k, energy, m, spin in data['extrema']:        
        ax.plot(xfit, yfit, '-')
        ax.plot(x, eigs[:,indices], 'o')

    ax.set_xlabel('k [Ang$^{-1}$]')
    ax.set_ylabel('e - e$_F$ [eV]')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()




if __name__ == '__main__':
    main.cli()







