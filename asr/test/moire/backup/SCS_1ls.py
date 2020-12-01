from asr.core import command, option

"""
structure: layer1.json/layer2.json
inputfile: layer1.gpw/layer2.gpw
"""

def BandStructure(structure, inputfile, bandpathpoints: int = 80)
    from gpaw import GPAW, PW, FermiDirac
    from ase.io import read
    from ase.dft.kpoints import bandpath
    
    atoms = read(structure)
    lat = atoms.cell.get_bravais_lattice()

    specpts = ase.dft.kpoints.get_special_points()
    path = ''
    for key in specpts:
        path += key
    
    bandpath = atoms.cell.bandpath(path, npoints=bandpathpoints)
    
    calc = GPAW(inputfile,
                fixdensity=True,
                symmetry='off',
                kpts=bandpath,
                txt='bs_lcao.txt')
    
    calc.get_potential_energy()
    calc.write('bs_lcao.gpw', 'all')
 

@command('asr.SCS_1ls')
@option('--structure', type = str)
@option('--kpoints', type = int)
@option('--eta', type = float)
def main(structure: str = None, kpoints: int = 18, eta: float = 0.01):
    '''
    This recipe uses the self-consistent scissors operator.
    OBS: Note that the SCS in defined in a development branch of GPAW 
    and has yet to be merged into the main version.
    '''
    import json
    from gpaw import GPAW, PW, FermiDirac
    from gpaw.lcao.scissors import Scissors
    from ase import Atoms
    from ase.io import read

    if structure == "layer1.json":
        gs_name = "layer1.gpw"
        layer_nr = 1
    elif structure == "layer2.json":
        gs_name = "layer2.gpw"
        layer_nr = 2
    else:
        raise AssertionError('Only use this recipe for the single layers in the SCS calc!')

    # Loading the structure
    atoms = read(structure)
    # Setup the calculator 
    calc = GPAW(mode='lcao',
            xc='PBE',
            basis='dzp',
            kpts=(kpoints, kpoints, 1),
            occupations=FermiDirac(eta),
            txt=f'asr.bilayers_scs_layer{layer_nr}.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(gs_name, 'all')


if __name__ == "__main__":
    main.cli()
