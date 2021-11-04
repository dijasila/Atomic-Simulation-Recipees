from ase.db import connect

name = "name_of_my_database"
title = "Title of my database."
database = connect("database.db")
key_descriptions = {
    "name_of_a_key": ("The short description", "The long description", "The units"),
    "id": ("ID", "Uniqe row ID", ""),
    "age": ("Age", "Time since creation", ""),
    "formula": ("Formula", "Chemical formula", ""),
    "pbc": ("PBC", "Periodic boundary conditions", ""),
    "user": ("Username", "Username", ""),
    "calculator": ("Calculator", "ASE-calculator name", ""),
    "energy": ("Energy", "Total energy", "eV"),
    "natoms": ("Number of atoms", "Number of atoms", ""),
    "fmax": ("Maximum force", "Maximum force", "eV/Ang"),
    "smax": ("Maximum stress", "Maximum stress on unit cell", "eV/Ang<sup>3</sup>"),
    "charge": ("Charge", "Net charge in unit cell", "|e|"),
    "mass": ("Mass", "Sum of atomic masses in unit cell", "au"),
    "magmom": ("Magnetic moment", "Magnetic moment", "au"),
    "unique_id": ("Unique ID", "Random (unique) ID", ""),
    "volume": ("Volume", "Volume of unit cell", "Ang<sup>3</sup>"),
}