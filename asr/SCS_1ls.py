from asr.core import command, option
import numpy as np
from numpy.linalg import norm


def AngleBetween(v1, v2):
    v1u = v1 / norm(v1)
    v2u = v2 / norm(v2)
    if v2u[1] >= 0 and v1u[1] >= 0:
        return np.arccos(v2u[0]) - np.arccos(v1u[0])
    if v2u[1] <= 0 and v1u[1] <= 0:
        return np.arccos(v1u[0]) - np.arccos(v2u[0])
    if v2u[1] >= 0 and v1u[1] <= 0:
        return np.arccos(v1u[0]) + np.arccos(v2u[0])
    if v2u[1] <= 0 and v1u[1] >= 0:
        return - np.arccos(v1u[0]) - np.arccos(v2u[0])


# Obtain symmetry and BZ path for the monolayer
def GetBandpath2D(atoms):
    tol_ang = np.pi / 180
    tol_norm = 1e-4
    cell = atoms.get_cell()
    angle = AngleBetween(cell[0], cell[1])
    norm0 = norm(cell[0])
    norm1 = norm(cell[1])
    # Hexagonal lattice
    if abs(angle - np.pi/3) < tol_ang or abs(angle - 2/3*np.pi) < tol_ang:
        if abs(norm0 - norm1) < tol_norm:
            return 'GMKG'
    # Square lattice
    elif abs(angle - np.pi/2) < tol_ang and abs(norm0 - norm1) < tol_norm:
        return 'MGXM'
    # Rectangular lattice
    elif abs(angle - np.pi/2) < tol_ang and abs(norm0 - norm1) > tol_norm:
        return 'GXSYGS'
    # Monoclinic lattice
    elif abs(angle - np.pi/2) > tol_ang and abs(angle - np.pi/3) > tol_ang and abs(angle - 2/3*np.pi) > tol_ang:
        return 'GYHCH1XG'


def SinglePoint(atoms, kpoints, eta, layer_nr): 
    from gpaw import GPAW, FermiDirac

    calc = GPAW(mode='lcao',
            xc='PBE',
            basis='dzp',
            kpts=(kpoints, kpoints, 1),
            occupations=FermiDirac(eta),
            txt=f'asr.bilayers_scs_layer{layer_nr}.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f'layer{layer_nr}.gpw', 'all')
 

def BandStructure(atoms, layer_nr, inputfile, bandpathpoints):
    from gpaw import GPAW
    from ase.io import read
    from ase.dft.kpoints import bandpath

    kptpath = atoms.cell.bandpath(path=GetBandpath2D(atoms), npoints=bandpathpoints, pbc=atoms.pbc)
    calc = GPAW(inputfile,
                fixdensity=True,
                symmetry='off',
                kpts=kptpath,
                txt=f'layer{layer_nr}_bandstructure.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f'layer{layer_nr}_bandstructure.gpw', 'all')


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
    spfile = f'layer{layer_nr}.gpw'
    bsfile = f'layer{layer_nr}_bandstructure.gpw'

    if Path(spfile).exists() == False:
        SinglePoint(atoms, kpoints, eta, layer_nr)

    if Path(bsfile).exists() == False:
        BandStructure(atoms, layer_nr, spfile, bandpathpoints) 


if __name__ == "__main__":
    main.cli()
