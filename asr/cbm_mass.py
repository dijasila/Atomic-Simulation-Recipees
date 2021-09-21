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

    data= row.data.get('results-asr.vbm_mass.json')

    extrematable = []
    for xfit, yfit, indices,x, eigs, k, energy, mass, spin  in data['extrema']:
        extrematable.append([f'{k:.3f}', f'{energy:.3f}', f'{mass:.3f}',f'{spin[0]:.1f},{spin[1]:.1f},{spin[2]:.1f}'])



    
    panel = WebPanel(describe_entry(f'cbm_mass', panel_description),
             columns=[[fig('cbm_mass.png')], []],
             plot_descriptions=[{'function': plot_cbm,
                                    'filenames': ['cbm_mass.png']}],
             sort=14)


    table = {'type': 'table',
             'header': ['Kpoint', 'Energy','Mass','Spin'],
             'rows': extrematable}



    panel2 = WebPanel(title= 'cbm_mass',
              columns= [[table]],
              sort=14)


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
    #print('data', len(data))
    
    for xfit, yfit, indices,x, eigs, k, energy, m, spin  in data['extrema']: 
        color = 0    
        for n in indices:
            plt.plot(x, eigs[:, n], 'o', color=f'C{color}')
            plt.plot(xfit, yfit, '-', color=f'C{color}')
            color += 1     
        #xfit = np.arange(0,4*np.pi,0.1)   # start,stop,step
        #yfit = np.sin(xfit) 

        #ax = plt.gca()
        plt.xlabel('k [Ang$^{-1}$]')
        plt.ylabel('e - e$_F$ [eV]')
        #ax.axis(ymin=0.0, ymax=1.3)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()


if __name__ == '__main__':
    main.cli()







