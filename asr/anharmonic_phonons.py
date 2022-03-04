# general python
import numpy as np
import typing
import os.path as path

# ase
from ase import Atoms
from ase.io import read, write
from ase.calculators.emt import EMT
from ase.parallel import world
from ase.build import make_supercell


# asr
from asr.core import (
    command,
    option,
    ASRResult,
    prepare_result)

# gpaw
from gpaw import GPAW, FermiDirac


def hiphive_fc23(
        atoms,
        cellsize,
        number_structures,
        rattle,
        mindistance,
        nat_dim,
        cut1,
        cut2,
        cut3,
        fc2n,
        fc3n,
        calculator):

    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy import Phonopy

    from hiphive.structure_generation import generate_mc_rattled_structures
    from hiphive.utilities import prepare_structures
    from hiphive import (ClusterSpace, StructureContainer,
                         ForceConstantPotential)
    from hiphive.fitting import Optimizer

    structures_fname = str(cellsize) + '_' + str(number_structures) + \
        '_' + str(rattle) + '_' + str(mindistance) + '.extxyz'

    # 2D or 3D calculation

    if nat_dim == 3:
        # 3D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0], [0, 0, cellsize]])
    else:
        # 2D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0], [0, 0, 1]])
    # calculator type

    atoms_ideal = make_supercell(atoms, multiplier)
    atoms_ideal.pbc = (1, 1, 1)
    print(atoms_ideal)
    if calculator == 'DFT':
        calc = GPAW(mode='lcao',
                    basis='dzp',
                    xc='PBE',
                    h=0.2,
                    kpts={"size": (2, 2, 1), "gamma": True},
                    symmetry={'point_group': False},
                    convergence={'forces': 1e-4},
                    occupations=FermiDirac(0.05),
                    txt='phono3py.txt')
    else:
        calc = EMT()
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
    # construct force constant potential
    fcp = ForceConstantPotential(cs, opt.parameters)

    # get phono3py supercell and build phonopy object. Done in series

    if world.rank == 0:
        atoms_phonopy = PhonopyAtoms(
            symbols=list(atoms.symbols),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.cell)
        phonopy = Phonopy(
            atoms_phonopy, supercell_matrix=multiplier,
            primitive_matrix=None)
        supercell = phonopy.get_supercell()
        supercell = Atoms(cell=supercell.cell, numbers=supercell.numbers, pbc=True,
                          scaled_positions=supercell.get_scaled_positions())

        # get force constants from fcp potentials
        fcs = fcp.get_force_constants(supercell)
        # write force constants
        fcs.write_to_phonopy(fc2n)
        fcs.write_to_phono3py(fc3n)


def phono3py_lifetime(atoms, cellsize, nat_dim, mesh_ph3,
                      fc2, fc3,
                      t1, t2, tstep):
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    # get phono3py supercell
    atoms_phonopy = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell)

    # 2D or 3D calculation
    if nat_dim == 3:
        # 3D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0],
                               [0, 0, cellsize]])
        meshnu = [mesh_ph3, mesh_ph3, mesh_ph3]
    else:
        # 2D calc
        multiplier = np.array([[cellsize, 0, 0], [0, cellsize, 0], [0, 0, 1]])
        meshnu = [mesh_ph3, mesh_ph3, 1]
    if world.rank == 0:

        # create phono3py object
        ph3 = Phono3py(
            atoms_phonopy,
            supercell_matrix=multiplier,
            primitive_matrix=None)
        ph3.mesh_numbers = meshnu
        ph3.set_fc3(fc3)
        ph3.set_fc2(fc2)

        # run thermal conductivity calculation
        ph3.set_phph_interaction()
        ph3.run_thermal_conductivity(
            temperatures=range(
                t1,
                t2,
                tstep),
            boundary_mfp=1e6,
            write_kappa=True)


@prepare_result
class Result(ASRResult):
    formats = {}
    #    formats = {"ase_webpanel": webpanel}

    temperature: float
    frequencies: typing.List[float]
    lifetimes: typing.List[float]
    kappa_00: float
    kappa_tensor: typing.List[float]

    key_descriptions = dict(
        kappa_00='Thermal conductivity',
        kappa_tensor='Kappa tensor',
        temperature='Temperature',
        frequencies='Frequencies',
        lifetimes='Lifetimes')


@command(
    "asr.anharmonic_phonons3_result",
    dependencies=[],
    requires=[
        "structure.json",
    ],
    returns=Result,
)
# @command('asr.anharmonic_phonons3')
@option('--atoms', type=str, default='structure.json')
@option("--cellsize", help="supercell multiplication for hiphive", type=int)
@option("--calculator", help="calculator type. DFT is the default", type=str)
@option("--rattle", help="rattle standard hiphive", type=float)
@option("--cut1", help="cutoff 2nd", type=float)
@option("--cut2", help="cutoff 3rd", type=float)
@option("--cut3", help="cutoff 4th", type=float)
@option("--nat_dim", help="spatial dimension number: 2D or 3D calculation", type=int)
@option("--mindistance", help="minimum distance hiphive", type=float)
@option("--number_structures", help="no. of structures rattle hiphive", type=int)
@option("--mesh_ph3", help="phono3py mesh", type=int)
@option("--t1", help="first temperature for thermal conductivity calculation", type=int)
@option("--t2", help="last temperature for thermal conductivity calculation", type=int)
@option("--tstep", help=" temperature step for thermal conductivity calculation",
        type=int)
def main(
        atoms: Atoms,
        cellsize: int = 6,
        calculator: str = 'DFT',
        rattle: float = 0.03,
        nat_dim: int = 2,
        cut1: float = 6.0,
        cut2: float = 5.0,
        cut3: float = 4.0,
        mindistance: float = 2.3,
        number_structures: int = 25,
        mesh_ph3: int = 20,
        t1=300,
        t2=301,
        tstep=1) -> Result:

    import h5py
    from phono3py.file_IO import read_fc3_from_hdf5, read_fc2_from_hdf5

    atoms = read(atoms)

    fc2n = 'fc2.hdf5'
    fc3n = 'fc3.hdf5'

    # call the two main functions

    hiphive_fc23(
        atoms,
        cellsize,
        number_structures,
        rattle,
        mindistance,
        nat_dim,
        cut1,
        cut2,
        cut3,
        fc2n,
        fc3n,
        calculator)

    # read force constants from hdf5
    fc3 = read_fc3_from_hdf5(filename=fc3n)
    fc2 = read_fc2_from_hdf5(filename=fc2n)

    phono3py_lifetime(atoms, cellsize, nat_dim, mesh_ph3, fc2, fc3,
                      t1, t2, tstep)

    # read the hdf5 file with the rta results

    phonopy_mesh = np.ones(3, int)
    phonopy_mesh[:nat_dim] = mesh_ph3

    label = ''.join(str(x) for x in phonopy_mesh)
    phonopy_outputfilename = f'kappa-m{label}.hdf5'

    with h5py.File(phonopy_outputfilename, 'r') as fd:
        temperatures = fd['temperature'][:]
        frequency = fd['frequency'][:]
        gamma = fd['gamma'][:]
        kappa = fd['kappa'][:]

    # write results to json file

    results = {
        "temperatures": temperatures,
        "frequency": frequency,
        "gamma": gamma,
        "kappa": kappa,
    }
    return results


if __name__ == "__main__":
    main.cli()
