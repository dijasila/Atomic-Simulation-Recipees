from ase.io import read, write
from ase.optimize import BFGS

from asr.core import command, option, ASRResult, read_json, prepare_result
from asr.relax import main as relax

from gpaw import GPAW, PW, restart
from gpaw.directmin.fdpw.directmin import DirectMin
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.exstatetools import excite_and_sort

from math import sqrt


@command('asr.excited')
@option('--excitation', type=str)
def calculate(excitation: str = 'alpha') -> ASRResult:

    atoms = read('../structure.json')

    old_calc = GPAW('../gs.gpw')

    old_params = old_calc.todict()

    charge = old_params['charge']

    calc = GPAW(mode=PW(600),
                xc='PBE',
                kpts={"size": (1,1,1), "gamma": True},
                spinpol=True,
                symmetry='off',
                eigensolver=DirectMin(),
                mixer={'name': 'dummy'},
                occupations={'name': 'fixed-uniform'},
                charge=charge,
                nbands='200%',
                maxiter=1000,
                txt='excited.txt'
                )

    f_sn = []
    for spin in range(olc_calc.get_number_of_spins()):
        f_n = [[0,1][e < ef] for e in olc_calc.get_eigenvalues(kpt=0, spin=spin)]
        f_sn.append(f_n)

    prepare_mom_calculation(calc, atoms, f_sn)

    atoms.calc = calc
    calc.initialize(atoms)
    atoms.get_potential_energy()
    write('ground.json', atoms)
 
    if excitation == 'alpha':
        excite_and_sort(calc.wfs, 0, 0, (0, 0), 'fdpw')
    if excitation == 'beta':
        excite_and_sort(calc.wfs, 0, 0, (1, 1), 'fdpw')

    calc.set(eigensolver=DirectMin(exstopt=True))

    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)

    prepare_mom_calculation(calc, atoms, f_sn)

    calculator = calc.todict()
    calculator['name'] = 'gpaw'

    BFGS(atoms,
         trajectory='relax.traj', 
         logfile='relax.log').run(fmax=0.01)

    write('structure.json', atoms)
    atoms.calc.write('excited.gpw')


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
