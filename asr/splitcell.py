from ase import Atoms
from ase.io import read, write
from pathlib import Path
from asr.core import command, option


@command('asr.splitcell')
@option('--sc_file', type = str)

def main(sc_file: str = "structure.json"):
    supercell = read(sc_file)
    layerlevels = supercell.get_tags()
    
    pos_a = []
    pos_b = []
    num_a = []
    num_b = []
    lvl_a = []
    lvl_b = []
    
    for i in range(len(layerlevels)):
        if layerlevels[i] == 1:
            pos_a.append(supercell.positions[i])
            num_a.append(supercell.numbers[i])
        else:
            pos_b.append(supercell.positions[i])
            num_b.append(supercell.numbers[i])
        
    layer_a = {}
    layer_b = {}
        
    layer_a["numbers"] = num_a
    layer_a["cell"] = supercell.cell
    layer_a["positions"] = pos_a
    layer_a["pbc"] = supercell.pbc
    atoms_a = Atoms.fromdict(layer_a)
    
    layer_b["numbers"] = num_b
    layer_b["cell"] = supercell.cell
    layer_b["positions"] = pos_b
    layer_b["pbc"] = supercell.pbc
    atoms_b = Atoms.fromdict(layer_b)
    
    write("layer1.json", atoms_a)
    write("layer2.json", atoms_b)


if __name__ == '__main__':
    main()









