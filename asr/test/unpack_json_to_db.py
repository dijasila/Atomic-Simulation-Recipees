from materials import std_test_materials
from asr.structureinfo import main as main_sif
from ase.io import write
import ase.db
import numpy as np



db = ase.db.connect("test_db.db")

test_struct = std_test_materials[1]
test_struct.set_initial_magnetic_moments(np.ones(2))



write("structure.json", test_struct)
data = main_sif()

data = data["magstate"]

print(data)

# db.write(test_struct, data)


# print(data)

