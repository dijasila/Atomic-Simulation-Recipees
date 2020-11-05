#general python
import numpy as np
import matplotlib.pyplot as plt
import h5py

#ase
from ase.io import *
from ase import Atoms
from ase.calculators.emt import EMT

#hiphive
from hiphive.structure_generation import generate_mc_rattled_structures
from hiphive.utilities import prepare_structures
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.fitting import Optimizer
from hiphive import ForceConstants

#asr
from asr.core import (command, option, DictStr, ASRResult,read_json, write_json, prepare_result,AtomsFile)

#phonopy & phono3py
from phonopy import Phonopy
from phono3py import Phono3py
from phonopy.structure.atoms import PhonopyAtoms
import phono3py
from phono3py.cui.create_force_constants import parse_forces, forces_in_dataset
from phono3py.file_IO import read_fc3_from_hdf5, read_fc2_from_hdf5
from phono3py.phonon3.fc3 import show_drift_fc3
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.interface.calculator import get_default_physical_units
import phonopy.cui.load_helper as load_helper


def hiphive_fc2(fname,fc2,fc3):
   #first read atoms object and generate structures
   structures_fname = 'rattled_structures.extxyz'

   # setup
   a=read(fname)
   atoms_ideal = Atoms(a).repeat(cell_size)
   calc = EMT()

   # generate structures
   structures = generate_mc_rattled_structures(atoms_ideal, n_structures, rattle_std, minimum_distance)
   structures = prepare_structures(structures, atoms_ideal, calc)

   # set up cluster space
   cutoffs = [5.0, 4.0, 4.0]
   cs = ClusterSpace(structures[0], cutoffs)

   # ... and structure container
   sc = StructureContainer(cs)
   for structure in structures:
       sc.add_structure(structure)

   # train model
   opt = Optimizer(sc.get_fit_data())
   opt.train()

   # construct force constant potential
   fcp = ForceConstantPotential(cs, opt.parameters)

   # get phono3py supercell and build phonopy object
   b=read(fname)
   prim = Atoms(b)
   atoms_phonopy = PhonopyAtoms(symbols=prim.get_chemical_symbols(),
                                scaled_positions=prim.get_scaled_positions(),
                                cell=prim.cell)
   phonopy = Phonopy(atoms_phonopy, supercell_matrix=dim*np.eye(3),
                     primitive_matrix=None)
   supercell = phonopy.get_supercell()
   supercell = Atoms(cell=supercell.cell, numbers=supercell.numbers, pbc=True,
                     scaled_positions=supercell.get_scaled_positions())

   #get force constants from fcp potentials
   fcs = fcp.get_force_constants(supercell)

   # write force constants 
   fcs.write_to_phonopy(fc2)
   fcs.write_to_phono3py(fc3)

def phono3py_lifetime(unit,fc2name,fc3name):

   # get phono3py supercell
   b=read(unit)
   prim = Atoms(b)
   atoms_phonopy = PhonopyAtoms(symbols=prim.get_chemical_symbols(),
                                scaled_positions=prim.get_scaled_positions(),
                                cell=prim.cell)

   # read force constants from hdf5
   fc3 = read_fc3_from_hdf5(filename=fc3name)
   fc2 = read_fc2_from_hdf5(filename=fc2name)

   #create phono3py object
   ph3=Phono3py(atoms_phonopy, supercell_matrix=dim*np.eye(3), mesh=mesh,  primitive_matrix=None)
   ph3.set_fc3(fc3)
   ph3.set_fc2(fc2)

   #run thermal conductivity calculation
   ph3.set_phph_interaction()
   ph3.run_thermal_conductivity(temperatures=range(t1, t2, tstep), boundary_mfp=1e6, write_kappa=True)


@command('asr.anharmonic_phonon')
@option('--atoms', type=str, default='structure.json')
#@option('--atoms', type=AtomsFile(), default='structure.json')
@option("--cellsize", help="supercell multiplication for hiphive", type=int)
@option("--rattle", help="rattle standard hiphive", type=float)
@option("--dim_phonopy", help="dimension in phonopy calculation", type=int)
@option("--mindistance", help="minimum distance hiphive", type=float)
@option("--number_structures", help="no. of structures rattle hiphive", type=int)
@option("--mesh_ph3", help="phono3py mesh", type=int)

@option("--temperature1", help="first temperature for thermal conductivity calculation", type=int)
@option("--temperature2", help="last temperature for thermal conductivity calculation", type=int)
@option("--temperature_delta", help=" temperature step for thermal conductivity calculation", type=int)

def main(atoms: Atoms, cellsize: int=8, rattle: float=0.03, dim_phonopy: int=5,  mindistance: float=2.3, number_structures: int=5,mesh_ph3: int=14,temperature1=300,temperature2=301,temperature_delta=1):

   #global variables
  
   global   n_structures
   global  cell_size
   global  rattle_std
   global  minimum_distance 
   global mesh
   global dim
   global t1
   global t2
   global tstep

   n_structures = number_structures
   cell_size = cellsize
   rattle_std = rattle
   minimum_distance = mindistance
   mesh = mesh_ph3
   dim=dim_phonopy
   mesh2=int(mesh/2)
   t1=temperature1
   t2=temperature2
   tstep=temperature_delta
   atomss=atoms   

   #internal fc filenames
   
   fc2n='fc2.hdf5'
   fc3n='fc3.hdf5'

   #call the two main functions

   hiphive_fc2(atomss,fc2n,fc3n)
   phono3py_lifetime(atomss,fc2n,fc3n)

   # read the hdf5 file with the rta results

   filename='kappa-m'+str(mesh2)+str(mesh2)+str(mesh2)+'.hdf5'
   f = h5py.File(filename,'r')
   temperatures = f['temperature'][:]
   frequency = f['frequency'][:]
   gamma = f['gamma'][:]
   kappa = f['kappa'][:]

   #write results to json file

   results = {
     "temperatures": temperatures,
     "frequency": frequency,
     "gamma": gamma,
      "kappa": kappa,
   }
   return results


   #PLOT LIFETIMES
   ms = 4;fs = 16
   plt.figure()
   plt.plot(frequency.flatten(), gamma[0].flatten(), 'o', ms=ms)
   plt.xlabel('Frequency (THz)', fontsize=fs)
   plt.ylabel('$\Gamma$ (THz)', fontsize=fs)
   plt.xlim(left=0)
   plt.ylim(bottom=0)
   plt.title('T={:d}K'.format(int(temperatures[0])))
   plt.gca().tick_params(labelsize=fs)
   plt.tight_layout()
   plt.savefig('lifetime.pdf')
   

if __name__ == "__main__":
    main.cli()
