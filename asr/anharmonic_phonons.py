"""Anharmonic phonon properties with hiphive and phono3py."""

# eneral python
import numpy as np
import typing
import os.path as path
import h5py

# ase
from ase import Atoms
from ase.io import read, write
from ase.parallel import world
from ase.build import make_supercell

# asr
from asr.core import (command,
                      DictStr,
                      option,
                      ASRResult,
                      prepare_result)


def calculate(calculator):
    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)
    print('calc', calc)
    return calc


def hiphive_fc23(atoms,
                 cellsize,
                 number_structures,
                 rattle,
                 mindistance,
                 nd,
                 cut1,
                 cut2,
                 cut3,
                 calculator):

    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy import Phonopy

    from hiphive.structure_generation import generate_mc_rattled_structures
    from hiphive.utilities import prepare_structures
    from hiphive import (ClusterSpace, StructureContainer,
                         ForceConstantPotential)
    from hiphive.fitting import Optimizer
    from hiphive import enforce_rotational_sum_rules

    from asr.core import read_json

    structures_fname = str(cellsize) + '_' + str(number_structures) + \
        '_' + str(rattle) + '_' + str(mindistance) + '.extxyz'

    # 2D or 3D calculation

    if nd == 3:
        # 3D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0], [0, 0, cellsize]])
    else:
        # 2D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0], [0, 0, 1]])

    atoms_ideal = make_supercell(atoms, multiplier)
    atoms_ideal.pbc = (1, 1, 1)
    if path.exists('params.json'):
        setup_params = read_json('params.json')
        myparams = setup_params['asr.gs']['calculator']
        calc = calculate(myparams)
    else:
        calc = calculate(calculator)

    # create rattled structures or read them from file

    if not path.exists(structures_fname):
        structures = generate_mc_rattled_structures(
            atoms_ideal, number_structures, rattle, mindistance)
        structures = prepare_structures(structures, atoms_ideal, calc)
        write(structures_fname, structures)
    else:
        structures = read(structures_fname + '@:')

    # proceed with cluster space generation and fcp optimization

    cutoffs = [cut1, cut2, cut3]
    cs = ClusterSpace(structures[0], cutoffs)
    sc = StructureContainer(cs)
    for structure in structures:
        sc.add_structure(structure)

    opt = Optimizer(sc.get_fit_data())
    opt.train()
    parameters = opt.parameters
    parameters_rot = enforce_rotational_sum_rules(cs,
                                                  parameters, ['Huang', 'Born-Huang'])
    fcp = ForceConstantPotential(cs, parameters_rot)
    print('fcp written')
    if world.rank == 0:
        atoms_phonopy = PhonopyAtoms(symbols=list(atoms.symbols),
                                     scaled_positions=atoms.get_scaled_positions(),
                                     cell=atoms.cell)
        phonopy = Phonopy(atoms_phonopy, supercell_matrix=multiplier,
                          primitive_matrix=None)
        supercell = phonopy.get_supercell()
        supercell = Atoms(cell=supercell.cell, numbers=supercell.numbers, pbc=True,
                          scaled_positions=supercell.get_scaled_positions())

        # get force constants from fcp potentials
        fcs = fcp.get_force_constants(supercell)
        # write force constants
        fcs.write_to_phonopy('fc2.hdf5')
        fcs.write_to_phono3py('fc3.hdf5')
    world.barrier()


def phono3py_lifetime(atoms, cellsize, nd, mesh_ph3,
                      t1, t2, tstep):
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    from phono3py.file_IO import read_fc3_from_hdf5, read_fc2_from_hdf5

    # get phono3py supercell
    atoms = read('structure.json')
    atoms_phonopy = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell)

    # 2D or 3D calculation
    if nd == 3:
        # 3D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0],
                               [0, 0, cellsize]])
        meshnu = [mesh_ph3, mesh_ph3, mesh_ph3]
    else:
        # 2D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0], [0, 0, 1]])
        meshnu = [mesh_ph3, mesh_ph3, 1]
    if world.rank == 0:

        # read force constants from hdf5
        fc3 = read_fc3_from_hdf5(filename='fc3.hdf5')
        fc2 = read_fc2_from_hdf5(filename='fc2.hdf5')

        # create phono3py object
        ph3 = Phono3py(atoms_phonopy,
                       supercell_matrix=multiplier,
                       primitive_matrix=None)
        ph3.mesh_numbers = meshnu
        ph3.set_fc3(fc3)
        ph3.set_fc2(fc2)

        # run thermal conductivity calculation
        ph3.set_phph_interaction()
        ph3.run_thermal_conductivity(temperatures=range(t1, t2, tstep),
                                     boundary_mfp=1e6,
                                     write_kappa=True)
    world.barrier()


@prepare_result
class Result(ASRResult):
    formats = {}
    #    formats = {"ase_webpanel": webpanel}

    temperatures: typing.List[float]
    frequencies: typing.List[float]
    gamma: typing.List[float]
    kappa: typing.List[float]
    heat_capacity: typing.List[float]
    group_velocity: typing.List[float]
    gv_by_gv: typing.List[float]
    mesh: typing.List[float]
    mode_kappa: typing.List[float]
    qpoint: typing.List[float]
    weight: typing.List[float]

    key_descriptions = dict(
        kappa='kappa',
        temperatures='temperature',
        frequencies='frequency',
        gamma='gamma',
        heat_capacity='heat_capacity',
        group_velocity='group_velocity',
        gv_by_gv='gv_by_gv',
        mesh='mesh',
        mode_kappa='mode_kappa',
        qpoint='qpoint',
        weight='weight')


@command(
    "asr.anharmonic_phonons_result",
    dependencies=[],
    requires=[
        "structure.json",
    ],
    returns=Result,
)
@option("--cellsize", help="supercell multiplication for hiphive", type=int)
@option("--calculator", help="calculator parameters. DFT is the default",
        type=DictStr())
@option("--rattle", help="rattle standard hiphive", type=float)
@option("--cut1", help="cutoff 2nd", type=float)
@option("--cut2", help="cutoff 3rd", type=float)
@option("--cut3", help="cutoff 4th", type=float)
@option("--nd", help="spatial dimension number: 2D or 3D calculation", type=int)
@option("--mindistance", help="minimum distance hiphive", type=float)
@option("--number_structures", help="no. of structures rattle hiphive", type=int)
@option("--mesh_ph3", help="phono3py mesh", type=int)
@option("--t1", help="first temperature for thermal conductivity calculation",
        type=float)
@option("--t2", help="last temperature for thermal conductivity calculation",
        type=float)
@option("--tstep", help=" temperature step for thermal conductivity calculation",
        type=float)
def main(cellsize: int = 5,
         calculator: dict = {'name': 'gpaw', 'mode': {'name': 'pw', 'ecut': 700},
                             'xc': 'PBE', 'basis': 'dzp',
                             'kpts': {'density': 8.0, 'gamma': True},
                             'occupations': {'name': 'fermi-dirac', 'width': 0.05},
                             'convergence': {'forces': 1e-6},
                             'txt': 'gs.txt', 'charge': 0},
         rattle: float = 0.03,
         nd: int = 2,
         cut1: float = 6.0,
         cut2: float = 5.0,
         cut3: float = 4.0,
         mindistance: float = 2.3,
         number_structures: int = 15,
         mesh_ph3: int = 20,
         t1: float = 0,
         t2: float = 1001,
         tstep: float = 10) -> Result:

    # call the two main functions

    atoms = read('structure.json')
    hiphive_fc23(atoms,
                 cellsize,
                 number_structures,
                 rattle,
                 mindistance,
                 nd,
                 cut1,
                 cut2,
                 cut3,
                 calculator)

    phono3py_lifetime(atoms, cellsize, nd, mesh_ph3,
                      t1, t2, tstep)

    # read the hdf5 file with the rta results

    phonopy_mesh = np.ones(3, int)
    phonopy_mesh[:nd] = mesh_ph3

    label = ''.join(str(x) for x in phonopy_mesh)
    phonopy_outputfilename = f'kappa-m{label}.hdf5'

    with h5py.File(phonopy_outputfilename, 'r') as fd:
        temperatures = fd['temperature'][:]
        frequencies = fd['frequency'][:]
        gamma = fd['gamma'][:]
        kappa = fd['kappa'][:]
        heat_capacity = fd['heat_capacity'][:]
        group_velocity = fd['group_velocity'][:]
        gv_by_gv = fd['gv_by_gv'][:]
        mesh = fd['mesh'][:]
        mode_kappa = fd['mode_kappa'][:]
        qpoint = fd['qpoint'][:]
        weight = fd['weight'][:]

    results = {"temperatures": temperatures,
               "frequencies": frequencies,
               "gamma": gamma,
               "kappa": kappa,
               "heat_capacity": heat_capacity,
               "group_velocity": group_velocity,
               "gv_by_gv": gv_by_gv,
               "mesh": mesh,
               "mode_kappa": mode_kappa,
               "qpoint": qpoint,
               "weight": weight}

    world.barrier()
    return results


if __name__ == "__main__":
    main.cli()
