from ase import Atoms
from ase.io import read, write
from ase.io.jsonio import read_json, write_json
import numpy as np
from pathlib import Path
import argparse


def json2atoms(jsonfile):
    dct = read_json(jsonfile)
    return Atoms.fromdict(dct)


def main(sc_file: str = "structure.json"):
    supercell = json2atoms("structure.json")
    layerlevels = supercell.info["layerlevels"]
    
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
            lvl_a.append(layerlevels[i])
        else:
            pos_b.append(supercell.positions[i])
            num_b.append(supercell.numbers[i])
            lvl_b.append(layerlevels[i])
        
    layer_a = {}
    layer_b = {}
        
    layer_a["numbers"] = num_a
    layer_a["cell"] = supercell.cell
    layer_a["positions"] = pos_a
    layer_a["pbc"] = supercell.pbc
    layer_a["info"] = {}
    layer_a["info"]["layerlevel"] = lvl_a
    atoms_a = Atoms.fromdict(layer_a)
    
    layer_b["numbers"] = num_b
    layer_b["cell"] = supercell.cell
    layer_b["positions"] = pos_b
    layer_b["pbc"] = supercell.pbc
    layer_b["info"] = {}
    layer_b["info"]["layerlevel"] = lvl_b
    atoms_b = Atoms.fromdict(layer_b)
    
    
    write_json("l1.json", layer_a)
    write("l1.xyz", atoms_a)
    
    write_json(f"l2.json", layer_b)
    write(f"l2.xyz", atoms_b)


if __name__ == '__main__':
    main()









