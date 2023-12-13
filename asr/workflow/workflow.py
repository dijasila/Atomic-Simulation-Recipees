from pathlib import Path
import os
from ase.io import read
from asr.core import read_json
from asr.workflow.bilayerutils import check_monolayer, has_tm3d, is_magnetic
from asr.workflow.bilayerutils import modify_params, gs_w_wo_U, starting_magmoms, analyse_w_wo_U, copy_gs_results, get_eb_interlayer, hubbardu_needed
from asr.workflow.bilayerutils import task, silentremove, listdirs, add2file, recently_updated, cleanup_gs, select_stables
from asr.stack_bilayer import BuildBilayer
from ase.db import connect
import numpy as np
import time
import shutil
from asr.dynstab import dynstab_manager

c2db = connect('/home/niflheim2/cmr/C2DB-ASR/collected-databases/c2db-first-class-20220908.db')

###########################
def getresources():
    """Randomly generate a resource.
    Use this to distribute jobs evenly across multiple partitions.
    """
    r = np.random.rand()
    if r<0.5:
        resources = "56:10h"
    elif r<1.6:
        resources = "40:10h"
    else:
        resources = "48:2d"

    #resources = "24:20h"
    resources = "24:2d"
    #resources = "56:50h"
    #resources = "24:10h"
    return resources


###########################
def all_done(list_of_tasks):
    """Determine if all tasks in list_of_tasks are done."""
    return all([task.check_creates_files()
                for task in list_of_tasks])


###########################
##### Bilayer tasks #######
###########################

###########################
def calc_vdw(folder):
    from ase.calculators.calculator import get_calculator_class

    if os.path.isfile(f"{folder}/vdw_e.npy"):
        silentremove(f"{folder}/vdwtask.*.*", checkotherfileexists=f'{folder}/vdw_e.npy')
        return

    Calculator = get_calculator_class("dftd3")
    calc = Calculator()

    atoms = read(f"{folder}/structure.json")
    atoms.set_calculator(calc)

    e = atoms.get_potential_energy()
    np.save(f"{folder}/vdw_e.npy", e)


###########################
def create_bilayers(folder, structure_file = "structure_initial.json", hundrule=False):
    """
    if magnetic info of the monolayer is not available one can use hund rule instead.
    """
    if hundrule:
        return [task(f"asr.stack_bilayer --hund-rule {hundrule}", creates = ['structure_adjusted.json','results-asr.stack_bilayer.json'], resources="1:10m", folder=folder)]
    else:
        return [task(f"asr.stack_bilayer", creates = ['structure_adjusted.json','results-asr.stack_bilayer.json'], resources="1:10m", folder=folder)]


###########################
def monolayer_treatment(folder, structure_file, initial_magmoms, e_threshold=1e-6, hubbardu=0):
    tasks = []
    res = getresources()
    tm3d = has_tm3d(folder, structure_file)
  
    if not tm3d and not os.path.isfile(f'{folder}/results-asr.gs.json'):
        # we copy the structure_adjusted, set initial magmoms and write as structure.json
        atoms = read(f'{folder}/{structure_file}')
        if initial_magmoms is not None:
            atoms.set_initial_magnetic_moments(initial_magmoms)
        atoms.write(f'{folder}/structure.json')

        mixer={'type':'sum','beta':0.05}
        modify_params(folder, etot_threshold=e_threshold, uvalue=0, mixer=mixer)

        tasks += [task("asr.gs", resources=res, folder=folder, restart=1)]
        calc_vdw(folder)

    elif tm3d:
        # we copy the structure_adjusted, check if we need U or not and write the correct one as structure.json
        mixer={'type':'sum','beta':0.02}

        subfolder_noU, subfolder_U = gs_w_wo_U(folder, 
                                               structure=structure_file, 
                                               initial_magmoms=initial_magmoms,
                                               hubbardu=hubbardu, 
                                               e_threshold=e_threshold,
                                               mixer=mixer)

        tasks += [task("asr.gs", resources=res, folder=subfolder_noU, restart=1)]
        calc_vdw(subfolder_noU)

        tasks += [task("asr.gs", resources=res, folder=subfolder_U, restart=1)]
        calc_vdw(subfolder_U)

        """
        Here based on the results with U we decide if we want to continue with or withoutU
        """
        analyse_w_wo_U(folder, subfolder_noU, subfolder_U)

    return tasks
 

###########################
def submit_zscan(blfolders, magnetic):
    res = getresources()
    tasks = []

    for blfolder in blfolders:
        ### Normally when the result file is there, myqueue should not submit a job but sometimes it does. To avoid it ...    
        if os.path.isfile(f'{blfolder}/results-asr.zscan.json'):
           continue

        ### If the structure is "shifted_from" and not magnetic: we use the previous interlayer to start the task.
        ### We don't do this for magnetic systems because we need them to start far and without hybridization
        distancearg = ""
        translation = read_json(f'{blfolder}/translation.json')
        if 'Shifted_from' in translation:
           original = translation['Shifted_from']
           resultsfname = f"{blfolder}/../{original}/results-asr.zscan.json"
           if os.path.exists(resultsfname) and not magnetic:
              h = read_json(resultsfname)['interlayer_distance']
              distancearg = f' --distance {h+0.2}'


        if not magnetic:
           if not os.path.isfile(f'{blfolder}/zscan.txt'):
               tasks += [task(f"asr.zscan{distancearg} --method optimize", resources=res,
                         creates=["results-asr.zscan.json"], restart=2, folder=blfolder)]
               continue
   
           if recently_updated(blfolder, 'zscan.txt'):
               continue 

        if not os.path.isfile(f'{blfolder}/results-asr.zscan.json'):
           tasks += [task(f"asr.zscan{distancearg} --method zscan", resources=res,
                            creates=["results-asr.zscan.json"], restart=2, folder=blfolder)]

    return tasks


###########################
def select_largest_ebs(blfolders, tm3d, deltaE, hubbardU=False, source='gs', ignore_failed=False):
    """
    This function selects the bilayers with largest Ebs within a window of deltaE
    """
    energies = []
    for subf in blfolders:
        zscan_result = f"{subf}/results-asr.zscan.json"

        # we sometime need to get the gs from a subfolder inside the bilayer 
        #subf = f'{blfolder}/{sub_folder}'

        # This part is for the time when we decide to ignore the bilayers that we could not finish zscan
        if ignore_failed and not os.path.isfile(zscan_result):
            continue

        # energies are in meV/A2
        eb, length = get_eb_interlayer(subf, hubbardU, source, tm3d)
        print(f"source={source}: {eb}, {length} : {subf}")
        if eb is None and ignore_failed:
           continue
        elif eb is None and not ignore_failed:
           print('>>> Zscan of some folders have not finished')
           return []

        energies.append((subf, eb))

    if len(energies)==0:
        print('>>> No bilayers found')
        return []
 
    cutoff = 150 
    maxE = max([e for s, e in energies])
    if maxE>cutoff:
        print('>>> Bilayer not exfoliable: ', subf)

    # Select materials within window of max
    selected = list(filter(lambda t: abs(t[1] - maxE) <= deltaE, energies))
    # Select materials that pass exfoliable criterion
    selected = list(filter(lambda t: t[1] <= cutoff, selected))
    selected = sorted(selected, key=lambda t: -t[1])[:]
    return [t[0] for t in selected]


#######################
def bilayer_gs(zscan_selected, structure_file, monolayer_folder, e_threshold, HubbardU=False, uvalue=0):
    """
    We want Eb without U. For consistancy we do this calculation inside a folder even if there is not TM3d atom
    If U is needed, and the structure is not magnetic, then we also make a gs-withU folder
    If the structure is magnetic, even if U is needed we will not create a gs-withU folder because the gs will be 
    done by the interlayer magnetic recipe in the main folder
    """
    tasks = []

    for bilayer_folder in zscan_selected:
        subfolder = f'{bilayer_folder}/gs-withU' if HubbardU else f'{bilayer_folder}/gs-withoutU'

        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
            # This will copy the zscan structure and initial magmoms will be same as bilayer prototype as zscan used it
            shutil.copy(f'{bilayer_folder}/{structure_file}',f'{subfolder}/structure.json')

        tasks += basic_properties(subfolder, monolayer_folder, HubbardU=HubbardU, uvalue=uvalue, afterJ=False, e_threshold=e_threshold, cleanup=False)            
    return tasks


#######################
def basic_properties(run_folder, monolayer_folder, HubbardU=False, uvalue=0, afterJ=False, e_threshold=1e-6, cleanup=False, symmetry=True):
    res = getresources()
    tasks = []

    magnetic = is_magnetic(monolayer_folder, c2db)

    # This function changes the params if the calculation failed before file and we do not want it to run if a gs calculation is running.
    if recently_updated(folder=run_folder, filename="gs.txt"):
        return []

    # If magnetic and we want to run in the main folder it has to be done afterJ (with correct magnetic order)
    if afterJ and magnetic:
       if not os.path.isfile(f'{run_folder}/results-asr.interlayer_magnetic_exchange.json'):
          return []

    #if not afterJ and not os.path.isfile(f'{run_folder}/results-asr.gs.json'):
    #   setup_fm(run_folder, monolayer_folder, HubbardU=HubbardU)
  
    # parameters of the params file
    mixer = {'type':'sum','beta':0.02} if magnetic else None
    uvalue_param = uvalue if HubbardU else None

    # I will put a params file in all the folders to: (1) Change the gs convergenve energy (2) setup the U values for the correct atoms.
    # we don't want to change the params file if the gs calculation is done with it 
    if not os.path.isfile(f'{run_folder}/params.json') and not os.path.isfile(f'{run_folder}/results-asr.gs.json'):
       modify_params(run_folder, etot_threshold=e_threshold, uvalue=uvalue_param, mixer=mixer, symmetry=symmetry)

    tasks += [task("asr.gs", resources=res, folder=run_folder, restart=2)]
    calc_vdw(run_folder)

    # Here I want to clean up a bit to save some space.
    if cleanup:
       cleanup_gs(run_folder)

    return tasks


###########################
def create_final_structure(blfolder, magnetic, uvalue, mlfolder=None):
    atoms = read(f'{blfolder}/structure_zscan.json')

    #res = getresources()
    if len(atoms)<13:
        res = "56:50h"
    elif len(atoms)<17:
        res = "112:50h"
    else:
        res = "224:50h"

    res = "56:50h"

    tasks = []


    # We don't want to redo this every time. So if the structure file is created we can ignore this function
    if os.path.isfile(f'{blfolder}/structure.json'):
        return tasks

    if mlfolder is None:
        mlfolder = f'{blfolder}/../'


    if not magnetic and not hubbardu_needed(mlfolder):
        # bring the gs without U in the main folder and create the structure file and gs.gpw
        copy_gs_results(origin=f'{blfolder}/gs-withoutU', destination=blfolder)


    elif not magnetic and hubbardu_needed(mlfolder):
        # bring the gs with U in the main folder and create the structure file and  gs.gpw
        copy_gs_results(origin=f'{blfolder}/gs-withU', destination=blfolder)


    elif magnetic and not hubbardu_needed(mlfolder) and not os.path.isfile(f'{blfolder}/results-asr.interlayer_magnetic_exchange.json'):
        # submit the interlayer magnetic with U=0 which will create the structure file and gs.gpw
        tasks += [task(f"asr.interlayer_magnetic_exchange -u 0.0", resources=res, folder=blfolder,
                 restart=2, creates="results-asr.interlayer_magnetic_exchange.json")]


    elif magnetic and hubbardu_needed(mlfolder) and not os.path.isfile(f'{blfolder}/results-asr.interlayer_magnetic_exchange.json'):
        # submit the interlayer magnetic with U=uvalue which will create the structure file and gs.gpw
        tasks += [task(f"asr.interlayer_magnetic_exchange -u {uvalue}", resources=res, folder=blfolder,
                 restart=2, creates="results-asr.interlayer_magnetic_exchange.json")]

    return tasks


###########################
def check_zscan(blfolders):
    from asr.zscan import min_quality
    for folder in blfolders:
        if os.path.isfile(f'{folder}/results-asr.zscan.json'):
            data = read_json(f'{folder}/results-asr.zscan.json')
            hh = data['heights']
            ee = data['energies']
            if not min_quality(energies=ee, heights=hh):
                print('>>>>> Problem in zscan >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
                add2file('/home/niflheim/sahpa/stacking-venv/asr/asr/workflow/zscan_problem.txt', folder)


###########################
##### Properties tasks ####
###########################

###########################
def bandstructure(bilayer_folder):
    res = getresources()
    # The gs task is not submitted in this folder but the result exists so we can not use the deps of mq
    if not os.path.isfile(f'{bilayer_folder}/results-asr.gs.json'):
        return []
    return [task("asr.bandstructure", resources=res,
                 folder=bilayer_folder, restart=2)]

##########################
def pdos(bilayer_folder):
    res = getresources()
    # The gs task is not submitted in this folder but the result exists so we can not use the deps of mq
    if not os.path.isfile(f'{bilayer_folder}/results-asr.gs.json'):
        return []
    return [task("asr.pdos", resources=res, folder=bilayer_folder)]


##########################
def projected_bandstructure(bilayer_folder):
    return [task("asr.projected_bandstructure", resources='1:10m',
            folder=bilayer_folder, restart=2, deps=['asr.bandstructure'])]


##########################
def fermisurface(bilayer_folder):
    res = getresources()
    # The gs task is not submitted in this folder but the result exists so we can not use the deps of mq
    if not os.path.isfile(f'{bilayer_folder}/results-asr.gs.json'):
        return []
    gap = read_json(f"{bilayer_folder}/results-asr.gs.json").get("gap")
    return [task("asr.fermisurface", resources="1:30m", folder=bilayer_folder, restart=2)] if (gap == 0.0) else []


##########################
def emasses(bilayer_folder):
    """Run emasses if material has a gap"""
    res = getresources()
    gap = read_json(f"{bilayer_folder}/results-asr.gs.json").get("gap")
    return [task("asr.emasses", resources=res, folder=bilayer_folder, restart=2)] if (gap > 0.01) else []


##########################
def hse(bilayer_folder):
    res = getresources()
    if not os.path.isfile(f"{bilayer_folder}/results-asr.gs.json") or not os.path.isfile(f'{bilayer_folder}/results-asr.bandstructure.json'):
       return []
    gap = read_json(f"{bilayer_folder}/results-asr.gs.json").get("gap")
    return [task("asr.hse", resources=res, folder=bilayer_folder, restart=2, deps=['asr.bandstructure','asr.gs'])] if (gap > 0.01) else []


##########################
def raman(bilayer_folder, magnetic):
    #res = getresources()
    tasks = []
    res = "56:50h"

    # only for non-magnetics
    if not magnetic and len(atoms)<17:
        tasks += [task("asr.ramanpol", resources=res, folder=bilayer_folder)]
    return tasks


##########################
def full_relax(mlfolder, gs_selected):
    """
    In this project we don't use U for structure relaxations
    """
    res = getresources()
    tasks = []
    # Monolayer full relaxation
    create_fullrelax_folder(mlfolder)
    tasks += [task("asr.relax --d3", creates=["results-asr.relax.json"],
              restart=2, resources=res, folder=f'{mlfolder}/fullrelax')]


    # Bilayer full relax for all the bilayers in the 3meV window
    for blfolder in gs_selected:
        create_fullrelax_folder(blfolder)
        tasks += [task("asr.relax --d3", creates=["results-asr.relax.json"],
                  restart=2, resources=res, folder=f'{blfolder}/fullrelax')]

    return tasks


###########################
##### Create tasks ########
###########################

class WorkflowControlParams:
    # Do not remove any item. Set it to None if you want to ignore it
    monolayer_criteria = {'max_natom': 10, 
                          'c2db': {'in_c2db' : True,
                                   'dynamic_stability_phonons': 'high',
                                   'dynamic_stability_stiffness': 'high',
                                   'thermodynamic_stability_level': 3,
                                   'ehull': 0.1,
                                  },
                          'has_tm3d': None,   # True/False/None
                         }
  

    forced_monolayers = ['P4-276f0a298324']
    hund_rule = False

    hubbardu = 4.0
    e_threshold = 1e-6
    e_threshold_dynstab = 1e-5
    deltaE = 3
    ignore_failed_zscan = True #False #True 
    ignore_failed_zscan_percentage = 20/100
    ignore_failed_gs = False


###########################
def create_tasks():
    tasks = []

    mlfolder = Path(".").absolute()
    print('\n\n>>>>> Monolayer: ', mlfolder)

    monolayer_criteria = WorkflowControlParams.monolayer_criteria
    hubbardu = WorkflowControlParams.hubbardu
    e_threshold = WorkflowControlParams.e_threshold
    ignore_failed_zscan = WorkflowControlParams.ignore_failed_zscan
    ignore_failed_zscan_percentage = WorkflowControlParams.ignore_failed_zscan_percentage
    ignore_failed_gs = WorkflowControlParams.ignore_failed_gs
    deltaE = WorkflowControlParams.deltaE
    

    hund_rule = WorkflowControlParams.hund_rule
    if 'hund' in str(mlfolder):
        hund_rule = True

    if not os.path.isfile(f'{mlfolder}/structure_initial.json'):
        raise FileNotFoundError(f'Monolayer "structure_initial.json" file not found >>> {mlfolder}')

    tm3d = has_tm3d(mlfolder, structure_file="structure_initial.json") 
    
    ml_magmoms  = starting_magmoms(c2db_structure_path=f'{mlfolder}/structure_initial.json', hundrule=hund_rule)

    mlfolder = check_monolayer(mlfolder,
                               criteria = monolayer_criteria, 
                               c2db = c2db,
                               structure_file = "structure_initial.json",
                               exceptions = WorkflowControlParams.forced_monolayers)

    if len(mlfolder)==0:
        return tasks

    tasks += create_bilayers(mlfolder, structure_file = "structure_initial.json", hundrule=hund_rule)
    if not all_done(tasks):
       return tasks

    tasks += monolayer_treatment(mlfolder, 
                                 structure_file = "structure_adjusted.json", 
                                 initial_magmoms = ml_magmoms,
                                 e_threshold=e_threshold, 
                                 hubbardu=hubbardu)
    if not all_done(tasks): 
       print('Monolayer gs with and without not finished yet', mlfolder)
       return tasks

    magnetic = is_magnetic(mlfolder=mlfolder, c2db=c2db)
    print(f">>> Monolayer is magnetic: {magnetic}")
    if magnetic is None:
       print("Monolayer gs is not done and we can not verify if magnetic")
       return tasks
    
    blfolders = [bl for bl in listdirs(mlfolder) if '-2-' in bl]
 
    check_zscan(blfolders)

    tasks += submit_zscan(blfolders, magnetic)

    if not all_done(tasks) and not ignore_failed_zscan:
       print('Not all zscan calculations done', mlfolder)
       return tasks

    elif not all_done(tasks) and ignore_failed_zscan:
       zscan_attempted = [bl for bl in blfolders if os.path.isfile(f'{bl}/zscan.txt') and not recently_updated(folder=bl, filename="zscan.txt")]
       zscan_not_done = [bl for bl in blfolders if not os.path.isfile(f'{bl}/results-asr.zscan.json')]
       if not len(blfolders)==len(zscan_attempted) or len(blfolders)*ignore_failed_zscan_percentage<=len(zscan_not_done):
          print("Not all zscan done. Either not all finished or many of them not converged.")
          return tasks
 
    zscan_selected = select_largest_ebs(blfolders, tm3d, deltaE=10, hubbardU=False, source='zscan', ignore_failed=ignore_failed_zscan)
    if len(zscan_selected)==0:
       print('Stopped because no bilayer was selected from zscan step.')
       return tasks

    tasks += bilayer_gs(zscan_selected, 
                        structure_file = "structure_zscan.json", 
                        monolayer_folder = mlfolder, 
                        e_threshold = e_threshold, 
                        HubbardU=False, 
                        uvalue=0)

    gs_selected = select_largest_ebs(zscan_selected, tm3d, deltaE=deltaE, hubbardU=False, source='gs', ignore_failed=ignore_failed_gs) ###0.005

    for blfolder in gs_selected:
        tasks += dynstab_manager(blfolder, 
                                 start_structure = 'structure_zscan.json',
                                 dynstab_path='DynStab', 
                                 step_num=2, 
                                 resources=getresources())


    stable_selected = select_stables(gs_selected)

    if hubbardu_needed(mlfolder) and not magnetic:
        tasks += bilayer_gs(zscan_selected, 
                            structure_file = "structure_zscan.json", 
                            monolayer_folder = mlfolder, 
                            e_threshold = e_threshold, 
                            HubbardU=True, 
                            uvalue=hubbardu)

    if magnetic:
        for blfolder in stable_selected:
            tasks += create_final_structure(blfolder, magnetic, hubbardu)

    else:
        for blfolder in zscan_selected:
            tasks += create_final_structure(blfolder, magnetic, hubbardu)

    for blfolder in stable_selected:

        print("Bilayer reached properties: ", blfolder)
        if not os.path.isfile(f'{blfolder}/structure.json') or not os.path.isfile(f'{blfolder}/gs.gpw'):
            continue

        tasks += bandstructure(blfolder)
        tasks += pdos(blfolder)
        tasks += raman(blfolder, magnetic)
        #tasks += projected_bandstructure(blfolder)
        tasks += fermisurface(blfolder)

    print()
    return tasks


if __name__ == "__main__":
    tasks = create_tasks()

    for ptask in tasks:
        print(ptask, ptask.is_done())
