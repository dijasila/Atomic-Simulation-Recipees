"""General purpose utilities."""
from datetime import datetime

_LATEST_PRINT = None


def timed_print(*args, wait=20):
    """Print at most every `wait` seconds."""
    global _LATEST_PRINT
    now = datetime.now()
    if _LATEST_PRINT is None or (now - _LATEST_PRINT).seconds > wait:
        print(*args)
        _LATEST_PRINT = now


def extract_structure(traj='relax.traj', tol=0.02, dest='structure.json'):
    """Extract the structure with lowest fmax from a trajectory file 
       if the lowest fmax is less than the requested tolerance,
       Then write it to file.
    """
    from ase.io import Trajectory
    import numpy as np

    def fmax(atoms):
        forces = atoms.get_forces()
        return np.sqrt((forces**2).sum(axis=1).max())
    
    traj = Trajectory(traj)
    forces = np.asarray([fmax(atoms) for atoms in traj])
    fmin = forces.min()
    if fmin <= tol:
        print(fmin)
        struct = traj[forces.argmin()]
        struct.write(dest)
    else:
        print(f'Lowest fmax found: {fmin}, larger than requested {tol}')
