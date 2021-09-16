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
   """vbm effektive mass plot""")

def webpanel(result, row, key_descriptions):
    from asr.database.browser import WebPanel

    
    panel = WebPanel(describe_entry(f'vbm_mass', panel_description),
             columns=[[fig('vbm_mass.png')], []],
             plot_descriptions=[{'function': plot_vbm,
                                    'filenames': ['vbm_mass.png']}],
             sort=3)

    return [panel]


from asr.core import command, option, ASRResult
@command('asr.vbm_mass')
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
                  kind='vbm') 
 
    
   
    results_dict={'extrema': extrema}
    print(results_dict)
    return Result(data=results_dict)


class Result(ASRResult):
    extrema:dict
    key_descriptions = {"vbm_mass" : "vbm effective mass"} 
    formats = {"ase_webpanel": webpanel}


def plot_vbm(row, fname):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    data= row.data.get('results-asr.vbm_mass.json') 
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







