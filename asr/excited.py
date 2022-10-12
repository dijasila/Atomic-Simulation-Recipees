from ase.io import read, write
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory

from asr.core import command, option, ASRResult, read_json, prepare_result
from asr.relax import main as relax

from gpaw import GPAW, PW, restart
from gpaw.directmin.fdpw.directmin import DirectMin
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.exstatetools import excite_and_sort
from pathlib import Path
from math import sqrt
import numpy as np

@command('asr.excited')
@option('--excitation', type=str)
def calculate(excitation: str = 'alpha' or 'beta') -> ASRResult:
    atoms = read('../structure.json')
    eref = atoms.get_potential_energy()
    ground_traj = Path('./ground.traj')
    magmoms = np.zeros(len(atoms))
    magmoms[0:len(atoms)]=2/len(atoms)
    atoms.set_initial_magnetic_moments(magmoms=magmoms)
    if ground_traj.is_file() and ground_traj.stat().st_size:
        ground = Trajectory('ground.traj')[-1]
        eground = ground.get_potential_energy()    
        atoms = ground
        atoms.set_initial_magnetic_moments(magmoms=magmoms)

    old_calc = GPAW('../gs.gpw')

    old_params = read_json('../params.json')
    charge = old_params['asr.relax']['calculator']['charge']

    calc = GPAW(mode=PW(800),
                xc='PBE',
                kpts={"size": (1,1,1), "gamma": True},
                spinpol=True,
                symmetry='off',
                eigensolver=DirectMin(convergelumo=True),
                mixer={'name': 'dummy'},
                occupations={'name': 'fixed-uniform'},
                charge=charge,
                nbands='101%',
                maxiter=5000,
                txt='excited.txt'
                )

    
    atoms.calc = calc

    BFGS(atoms, trajectory='ground.traj', logfile='ground.log').run(fmax=0.01)
    
    write('ground.json', atoms)
    energy = atoms.get_potential_energy()

    params = read_json('params.json')
    excitation=params['asr.excited@calculate']['excitation']
    if excitation == 'alpha':
        excite_and_sort(calc.wfs, 0, 0, (0, 0), 'fdpw')
    if excitation == 'beta':
        excite_and_sort(calc.wfs, 0, 0, (1, 1), 'fdpw')
    
    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)
    
    calc.set(eigensolver=DirectMin(exstopt=True))
    prepare_mom_calculation(calc, atoms, f_sn, use_projections=True, use_fixed_occupations=False)
    
    excited_traj = Path('./excited.traj')

    BFGS(atoms, trajectory='excited.traj', logfile='excited.log').run(fmax=0.01)

    write('structure.json', atoms)
    try:
        atoms.calc.write('gs.gpw')
    except:
        pass

@prepare_result
class ExcitedResults(ASRResult):
    """Container for excited results."""

    zpl: float
    delta_Q: float

    key_descriptions = dict(
        zpl='Zero phonon line energy.',
        delta_Q='Displacement from the ground state.')


@command('asr.excited',
         dependencies=["asr.excited@calculate"])
def main() -> ExcitedResults:

    ground = read('ground.json')
    excited = read('structure.json')

    energy_ground = ground.get_potential_energy()
    energy_excited = excited.get_potential_energy()

    zpl = energy_excited - energy_ground

    # define the 1D coordinate
    m_a = ground.get_masses()
    delta_R = excited.positions - ground.positions
    delta_Q = sqrt(((delta_R**2).sum(axis=-1) * m_a).sum())

    return ExcitedResults.fromdata(
        delta_Q=delta_Q,
        zpl=zpl)


def webpanel(result, row, key_descriptions):
    from asr.browser import fig, table

    if 'something' not in row.data:
        return None, []

    table1 = table(row,
                   'Property',
                   ['something'],
                   kd=key_descriptions)
    panel = ('Title',
             [[fig('something.png'), table1]])
    things = [(create_plot, ['something.png'])]
    return panel, things


def create_plot(row, fname):
    import matplotlib.pyplot as plt

    data = row.data.something
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data.things)
    plt.savefig(fname)


group = 'property'
creates = ['something.json']  # what files are created
dependencies = []  # no dependencies
resources = '1:10m'  # 1 core for 10 minutes
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()
