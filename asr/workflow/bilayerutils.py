import os
from ase.io import read
from asr.core import read_json
import shutil
from asr.core import read_json, write_json
import glob
import time
import numpy as np


PARAMS = {
 "asr.gs@calculate": {
  "calculator": {
   "mixer": {
    "method": "sum",
    "beta": 0.02,
    "history": 3,
    "weight": 50
   },
   "setups": {
    "V": ":d,3.1",
    "Cr": ":d,3.5",
    "Mn": ":d,3.8",
    "Fe": ":d,4.0",
    "Co": ":d,3.3",
    "Ni": ":d,6.4",
    "Cu": ":d,4.0"
   },
   "name": "gpaw",
   "mode": {
    "name": "pw",
    "ecut": 800
   },
   "xc": "PBE",
   "basis": "dzp",
   "kpts": {
    "density": 12.0,
    "gamma": True
   },
   "occupations": {
    "name": "fermi-dirac",
    "width": 0.05
   },
   "convergence": {
    "energy" : {"name": "energy", "tol": 0.00001, "relative": False,  "n_old": 4},
    "bands": "CBM+3.0",
    "density": 0.000001
   },
   "symmetry": {},
   "nbands": "200%",
   "txt": "gs.txt",
   "maxiter": 7000,
   "charge": 0
  }
 }
}


###########################
def listdirs(path):
    if type(path) != str:
        path = str(path.absolute())
    return [f"{path}/" + x for x in os.listdir(path) if os.path.isdir(f"{path}/{x}")]


###########################
# This function is useful for analysis and not used in the workflow or recipes
def parse_folders():
    from argparse import ArgumentParser
    from pathlib import Path
    parser = ArgumentParser()
    parser.add_argument("folders", nargs="*", help="Monolayer folders to analyse.")
    args = parser.parse_args()

    if len(args.folders) > 0:
        folders = [Path(x).absolute() for x in args.folders]
    else:
        folders = [Path(".").absolute()]

    return folders


###########################
def task(*args, **kwargs):
    """Get MyQueue task instance."""
    from myqueue.task import task as mqtask

    name = kwargs.get("name") or args[0]
    if "creates" not in kwargs:
        kwargs["creates"] = [f"results-{name}.json"]
    return mqtask(*args, **kwargs)


###########################
def silentremove(filepath, checkotherfileexists=None):
    """ 
    Removes file if it exist. 
    Can check and remove only if another filer exists
    """
    for fname in glob.glob(filepath):
        if checkotherfileexists == None:
            if os.path.isfile(fname):
                os.remove(fname)
        elif os.path.isfile(checkotherfileexists):
            if os.path.isfile(fname):
                os.remove(fname)


###########################
def cleanup_gs(run_folder):
    files2remove = ['results-asr.magnetic_anisotropy.json', 'results-asr.magnetic_anisotropy.json',
                    'gs.gpw', 'asr.gs.*.err', 'asr.gs.*.out']

    for f in files2remove:
        silentremove(f'{run_folder}/{f}',checkotherfileexists=f'{run_folder}/results-asr.gs.json')

    silentremove(f"{run_folder}/vdwtask.*.out", checkotherfileexists=f'{run_folder}/vdw_e.npy')
    silentremove(f"{run_folder}/vdwtask.*.err", checkotherfileexists=f'{run_folder}/vdw_e.npy')


###########################
def add2file(filename, item):
    if os.path.isfile(filename):
       text_file = open(filename, "r")
       lines = text_file.readlines()
    else:
       text_file = open(filename, "w")
       lines = []
    text_file.close()
    items = []
    for line in lines:
        line = line[:-1]
        items.append(line)

    textfile2 = open(filename, "a")
    if item not in items:
       textfile2.write(item + "\n")
    textfile2.close()


###########################
def recently_updated(folder, filename, time_in_seconds=3600):
    recent_files = [fle for rt, _, f in os.walk(folder) for fle in f if time.time() - os.stat(os.path.join(rt, fle)).st_mtime < time_in_seconds]
    return (filename in recent_files)
            

###########################
def set_info(folder, key, value, filename='info.json'):
    from pathlib import Path
    infofile = Path(f'{folder}/{filename}')

    if infofile.is_file():
        info = read_json(infofile)
    else:
        info = {}

    info[key] = value
    write_json(infofile, info)


###########################
def save_params_file(params_file, search_list, replace_list):
    """ opens the file, replaces the items in the search_list 
        with items in the replace_list"""
    with open(params_file, 'r') as file:
        data = file.read()
        for old, new in zip(search_list, replace_list):
            data = data.replace(old, new)
    with open(params_file, 'w') as file:
        file.write(data)


###########################
def modify_params(folder, etot_threshold=None, uvalue=None, mixer=None, symmetry=True):
    """ This function can become a class."""
    params_file = f'{folder}/params.json'
    write_json(params_file, PARAMS)

    # correct energy convergence
    if etot_threshold is not None:
        search_list = ['"energy" : {"name": "energy", "tol": 0.00001, "relative": false,  "n_old": 4},']
        replace_list = ['"energy" : {"name": "energy", "tol": '+str(etot_threshold)+', "relative": false, "n_old": 3},']
        save_params_file(params_file, search_list, replace_list)

    # correct mixer
    if mixer is not None:
        search_list = ['"method": "sum",',
                       '"beta": 0.02,']
        replace_list = [f'"method": "{mixer["type"]}",',
                        f'"beta": {mixer["beta"]},']
        save_params_file(params_file, search_list, replace_list)

    # correct symmetry: sometimes you may want to turn symmetries off
    if not symmetry:
        search_list = ['"symmetry": {},']
        replace_list = ['"symmetry": "off",']
        save_params_file(params_file, search_list, replace_list)


    # correct hubbard u: Negative u value means we want to use the matrials project list
    with open(params_file, 'r') as myfile:
        search_list = [line for line in myfile if ":d," in line]
    if uvalue is not None and uvalue!=0:
        if uvalue>0:
            replace_list = [f'    "V": ":d,{uvalue}",\n',
                           f'    "Cr": ":d,{uvalue}",\n',
                           f'    "Mn": ":d,{uvalue}",\n',
                           f'    "Fe": ":d,{uvalue}",\n',
                           f'    "Co": ":d,{uvalue}",\n',
                           f'    "Ni": ":d,{uvalue}",\n',
                           f'    "Cu": ":d,{uvalue}"\n']
        elif uvalue<0:
            replace_list = [f'    "V": ":d,3.1",\n',
                            f'    "Cr": ":d,3.5",\n',
                            f'    "Mn": ":d,3.8",\n',
                            f'    "Fe": ":d,4.0",\n',
                            f'    "Co": ":d,3.3",\n',
                            f'    "Ni": ":d,6.4",\n',
                            f'    "Cu": ":d,4.0"\n']
    elif uvalue is None or uvalue==0:
        replace_list = ['','','','','','','']

    save_params_file(params_file, search_list, replace_list)


###########################
def has_tm3d(folder, structure_file="structure.json"):
    contains_tm3d = False
    TM3d_atoms = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
    try:
        atoms = read(f"{folder}/{structure_file}")
    except:
        atoms = read(f"{folder}/structure_zscan.json")
    return any(x in atoms.symbols for x in TM3d_atoms)


###########################
def is_magnetic(mlfolder, c2db):
    '''
    We have the uid from the finger print of c2db 
    but if the uid changes that is also not useful
    '''
    gs = f'{mlfolder}/gs.gpw'

    try:
        monolayer_uid = mlfolder.split('/')[-1]
        for row in c2db.select(f"uid={monolayer_uid}"):
            return row.magstate!='NM'
    except:
        if os.path.isfile(gs):
            magmoms = read(f'{folder}/gs.gpw').get_magnetic_moments()    
            return len( [i for i in magmoms if abs(i) >= 0.01] )>0
        else:
            return None


###########################
def get_eb_interlayer(blfolder, hubbardU, source, tm3d, mlfolder=None):

    if mlfolder is None:
        mlfolder = f"{blfolder}/../"

    cell = read(f"{mlfolder}/structure.json").cell   
    cellarea = np.linalg.norm(np.cross(cell[0], cell[1]))

    # Without interlayer we have neither
    zscan_result = f"{blfolder}/results-asr.zscan.json"
    if os.path.isfile(zscan_result):
        data = read_json(zscan_result)
    else:
        data = {"zscan_binding_energy": None,
                "interlayer_distance": None}

    if source == 'zscan':
        return data["zscan_binding_energy"], data["interlayer_distance"]

    if source == 'gs':
        try:
            if not tm3d:
               ebl =read_json(f"{blfolder}/gs-withoutU/results-asr.gs.json")['etot']+np.load(f'{blfolder}/gs-withoutU/vdw_e.npy')
               eml = read_json(f"{mlfolder}/results-asr.gs.json")['etot']+np.load(f'{mlfolder}/vdw_e.npy')

            elif tm3d and abs(hubbardU)<0.01:
                ebl =read_json(f"{blfolder}/gs-withoutU/results-asr.gs.json")['etot']+np.load(f'{blfolder}/gs-withoutU/vdw_e.npy')
                eml = read_json(f"{mlfolder}/gs-withoutU/results-asr.gs.json")['etot']+np.load(f'{mlfolder}/gs-withoutU/vdw_e.npy')

            elif tm3d and abs(hubbardU)>=0.01:
                ebl = read_json(f"{blfolder}/gs-withU/results-asr.gs.json")['etot']+np.load(f'{blfolder}/gs-withU/vdw_e.npy')
                eml = read_json(f"{mlfolder}/gs-withU/results-asr.gs.json")['etot']+np.load(f'{mlfolder}/gs-withU/vdw_e.npy')

            eb  = -1000*(ebl-2*eml)/cellarea
            return eb, data["interlayer_distance"]
        except:
            return None, data["interlayer_distance"]


###########################
def check_monolayer(mlfolder, criteria=[], c2db=None, structure_file="structure_initial.json", exceptions=[]):
    '''This function is to help filter monolayers as we want.'''
    atoms = read(f"{mlfolder}/{structure_file}")

    """
    We want to add some monolayers ignoring the criteria we let them here
    """
    if str(mlfolder).split('/')[-1] in exceptions:
        return str(mlfolder)


    # This function is very locally defined to avoid repeatitions
    def in_crtiria(itemlist, item):
        if item in itemlist:
            if itemlist[item] is not None:
                return True
        return False

    # Number of atoms condition
    if in_crtiria(criteria, 'max_natom') and len(atoms)>criteria['max_natom']:
        print(f'Monolayer NOT accepted: Number of atoms more than {criteria["max_natom"]}', mlfolder)
        return []

    # Conditions applied if the matterial has to be in c2db and have certain properties
    if in_crtiria(criteria, 'c2db') and criteria['c2db']:

        monolayer_uid = str(mlfolder).split('/')[-1]

        if c2db is None:
            raise ValueError('Not connected to c2db')

        exists_in_c2db = False
        for c2db_row in c2db.select(f"uid={monolayer_uid}"):
            exists_in_c2db = True
             
            if in_crtiria(criteria['c2db'], 'dynamic_stability_phonons'):
                if not c2db_row.dynamic_stability_phonons == criteria['c2db']['dynamic_stability_phonons']:
                    print('Monolayer NOT accepted: Dynamical stability phonons', mlfolder)
                    return []

            if in_crtiria(criteria['c2db'], 'dynamic_stability_stiffness'):
                if not c2db_row.dynamic_stability_phonons == criteria['c2db']['dynamic_stability_stiffness']:
                    print('Monolayer NOT accepted: Dynamical stability stiffness', mlfolder)
                    return []

            if in_crtiria(criteria['c2db'], 'thermodynamic_stability_level'):
                if not c2db_row.thermodynamic_stability_level == criteria['c2db']['thermodynamic_stability_level']:
                    print('Monolayer NOT accepted: Thermodynamic stability level', mlfolder)
                    return []

            if in_crtiria(criteria['c2db'], 'ehull'):
                if not c2db_row.ehull <= criteria['c2db']['ehull']:
                    print('Monolayer NOT accepted: Thermodynamic stability not less than ehull limit', mlfolder)
                    return [] 

        if not exists_in_c2db:
            print('Monolayer NOT accepted: Not found in the c2db version provided', mlfolder)
            return []

    if in_crtiria(criteria, 'has_tm3d') and not tm3d(mlfolder):
        print(f'Monolayer NOT accepted: has_tm3d = {tm3d(mlfolder)}', mlfolder)
        return []

    return str(mlfolder)


###########################
def starting_magmoms(c2db_structure_path, hundrule=False):
    if hundrule:
        atoms = read(c2db_structure_path)
        TM3d_list = {'V': 1, 'Cr': 3, 'Mn': 5,'Fe': 4, 'Co': 3,'Ni': 2, 'Cu': 1}
        magmoms = [TM3d_list[atom.symbol] if atom.symbol in TM3d_list.keys() else 0 for atom in atoms]
    else:
        try:
            magmoms = read(c2db_structure_path).get_magnetic_moments()
        except:
            magmoms = None

    return magmoms


###########################
def gs_w_wo_U(folder, structure, initial_magmoms, hubbardu, e_threshold=None, mixer=None):

    if has_tm3d(folder, structure_file=structure):
        subfolder_noU = f'{folder}/gs-withoutU'
        subfolder_U = f'{folder}/gs-withU'

        for subfolder, uval in zip([subfolder_noU, subfolder_U], [0, hubbardu]):
            if not os.path.isdir(subfolder):
                os.mkdir(subfolder)
                atoms = read(f'{folder}/{structure}')
                if initial_magmoms is not None:
                    atoms.set_initial_magnetic_moments(initial_magmoms)
                atoms.write(f'{subfolder}/structure.json')
                # I will put a params file in all the folders to: (1) Change the gs convergenve energy (2) change mixer
                if 'withU' in subfolder:
                    modify_params(subfolder, etot_threshold=e_threshold, uvalue=hubbardu, mixer=mixer)
                else:
                    modify_params(subfolder, etot_threshold=e_threshold, uvalue=0.0, mixer=mixer)

        return subfolder_noU, subfolder_U


###########################
def analyse_w_wo_U(mlfolder, subfolder_noU, subfolder_U):
    # We set chosen_gs as empty so as long as it is empty we don't delete results in the subfolders
    chosen_gs = ""

    # We will check if monolayer is metalic with U then we do not consider U for it:
    if os.path.isfile(f'{subfolder_U}/results-asr.gs.json') and os.path.isfile(f'{subfolder_noU}/results-asr.gs.json'):

        chosen_gs = subfolder_U if hubbardu_needed(mlfolder) else subfolder_noU

        copy_gs_results(origin=chosen_gs, destination=mlfolder)

    # Clean up the subfolders to save space
    if chosen_gs != "" and os.path.isfile(f"{mlfolder}/gs.gpw"):
        silentremove(f'{subfolder_noU}/gs.gpw', checkotherfileexists=f'{mlfolder}/gs.gpw')
        silentremove(f'{subfolder_U}/gs.gpw', checkotherfileexists=f'{mlfolder}/gs.gpw')


###########################
def copy_gs_results(origin, destination):
        
    if os.path.isfile(f"{origin}/gs.gpw"):

        # The reason we don't just copy structure file is that we want to write the calculator on the structure
        new_structure = read(f'{origin}/gs.gpw')
        new_structure.write(f'{destination}/structure.json')

        files_to_copy = ["results-asr.gs.json", "gs.txt", "gs.gpw", "params.json",
                         "results-asr.structureinfo.json", "results-asr.magnetic_anisotropy.json",
                         "results-asr.magstate.json", "vdw_e.npy"]

        for filename in files_to_copy:
            shutil.copy(f"{origin}/{filename}", f"{destination}/{filename}")

        silentremove(f'{origin}/gs.gpw', checkotherfileexists=f'{destination}/gs.gpw')


###########################
def hubbardu_needed(mlfolder):
    if os.path.isfile(f'{mlfolder}/gs-withU/results-asr.gs.json'):
        gap = read_json(f"{mlfolder}/gs-withU/results-asr.gs.json").get("gap")
        return gap > 0.010


###########################
def check_stability(bl):
    result_file1 = f"{bl}/results-asr.dynstab.json"
    result_file2 = f"{bl}/dynstab.json"

    if os.path.isfile(result_file1):
        return read_json(result_file1)["status"]    
    elif os.path.isfile(result_file2):
        return read_json(result_file2)["DynStab_Result"]  
    else:
        return 'None' 

 
def check_status(bilayer_folders):
    return [check_stability(bilayer) for bilayer in bilayer_folders]


def select_stables(bilayer_folders):
    return [bilayer for bilayer in bilayer_folders if 'Stable' in check_stability(bilayer)]


###########################
def create_fullrelax_folder(folder):
    if not os.path.isdir(f"{folder}/fullrelax"):   
        os.mkdir(f"{folder}/fullrelax")

    if not os.path.isfile(f'{folder}/fullrelax/unrelaxed.json'):
        atoms = read(f"{folder}/structure.json")
        atoms.write(f'{folder}/fullrelax/unrelaxed.json')













