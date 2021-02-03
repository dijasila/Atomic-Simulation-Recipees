from ase.io.trajectory import Trajectory
from ase.io import write
from shutil import copy
import datetime
from asr.core import command, option

@command('asr.lastfromtraj')
@option('--file', type=str)
def main(file: str='unrelaxed.json'):
    now = datetime.datetime.now()
    datestr = now.strftime('%d%m%y%H%M')
    copy('unrelaxed.json', f'unrelaxed-{datestr}.json')
    
    traj = Trajectory('relax.traj')
    atoms = traj[-1]
    write('unrelaxed.json', atoms)

if __name__ == '__main__':
    main()

