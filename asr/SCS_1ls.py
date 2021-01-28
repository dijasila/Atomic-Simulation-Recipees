from asr.core import command, option
import numpy as np
from numpy.linalg import norm


def gs(atoms, kpoints, eta, layer_nr): 
    from gpaw import GPAW, FermiDirac

    calc = GPAW(mode='lcao',
            xc='PBE',
            basis='dzp',
            kpts=(kpoints, kpoints, 1),
            occupations=FermiDirac(eta),
            txt=f'layer{layer_nr}_gs.txt')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f'layer{layer_nr}_gs.gpw', 'all')
 

def bs(atoms, layer_nr, inputfile, bandpathpoints):
    from gpaw import GPAW
    from ase.dft.kpoints import bandpath

    kptpath = atoms.cell.bandpath(npoints=bandpathpoints, 
                                  pbc=atoms.pbc, 
                                  eps = 1e-2)

    calc = GPAW(inputfile,
                fixdensity=True,
                symmetry='off',
                kpts=kptpath,
                txt=f'layer{layer_nr}_bs.txt')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f'layer{layer_nr}_bs.gpw', 'all')


@command('asr.SCS_1ls')
@option('--structure', type = str)
@option('--kpoints', type = int)
@option('--eta', type = float)
@option('--bandpathpoints', type = int)
def main(structure: str = None,
         kpoints: int = 18, 
         eta: float = 0.01, 
         bandpathpoints: int = 80):
    '''
    This recipe uses the self-consistent scissors operator.
    OBS: Note that the SCS in defined in a development branch of GPAW 
    and has yet to be merged into the main version.
    '''
    from ase.io import read
    from pathlib import Path

    if structure == "layer1.json":
        layer_nr = 1
    elif structure == "layer2.json":
        layer_nr = 2
    else:
        raise AssertionError('Only use this recipe for the single layers in the SCS calc!')

    # Loading the structure
    atoms = read(structure)
    gsfile = f'layer{layer_nr}_gs.gpw'
    bsfile = f'layer{layer_nr}_bs.gpw'

    gs(atoms, kpoints, eta, layer_nr)
    bs(atoms, layer_nr, gsfile, bandpathpoints) 


if __name__ == "__main__":
    main.cli()
