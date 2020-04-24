import pytest
from ase.build import mx2
from .materials import std_test_materials


@pytest.fixture()
def duplicates_test_material():
    return std_test_materials[1]


@pytest.fixture()
def duplicates_test_db(asr_tmpdir):
    import ase.db
    import numpy as np
    from ase.io import write
    from asr.structureinfo import main
    from gpaw import GPAW


    db = ase.db.connect("db.db")
    
    # Setting up the reference database
    
    non_mag_ref = std_test_materials[1]
    data_non_mag = {"magstate": "NM"}
    db.write(non_mag_ref, data_non_mag) 
    
    
    mag_ref = std_test_materials[1] 
    data_mag = {"magstate": "FM"}
    db.write(mag_ref, data_mag) 
   

    supercell_ref = std_test_materials[1].repeat((2,2,2))
    db.write(supercell_ref, data_non_mag)

    return db 


@pytest.mark.ci
def test_duplicates(duplicates_test_material, duplicates_test_db, asr_tmpdir):
    from asr.duplicates import main
    from asr.structureinfo import main as main_struct_info
    from ase.io import write
    from asr.core import read_json 
    
    
    
    write("structure.json", duplicates_test_material)
    main_struct_info()
   
   
    results = main()
    

    assert results["duplicate"]
    assert results["duplicate_IDs"] == [1, 3]



@pytest.mark.ci
def test_duplicates_fm(duplicates_test_material, duplicates_test_db, asr_tmpdir):
    from asr.duplicates import main
    from asr.structureinfo import main as main_struct_info
    from ase.io import write
    from asr.core import read_json, write_json 
    
    
    write("structure.json", duplicates_test_material)
    data = main_struct_info()
    
    data["magstate"] = "FM"
    write_json("results-asr.structureinfo.json", data) 
    
    
    results = main()
    

    assert results["duplicate"]
    assert results["duplicate_IDs"] == [2]

