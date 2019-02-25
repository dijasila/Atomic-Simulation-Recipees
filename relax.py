import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read, Trajectory
from ase.io.formats import UnknownFileTypeError
from ase.io.ulm import open as ulmopen
from ase.io.ulm import InvalidULMFileError
from ase.parallel import world, broadcast
from ase.build import niggli_reduce
from gpaw import GPAW, PW, FermiDirac, KohnShamConvergenceError

from c2db import readinfo, magnetic_atoms
from c2db.bfgs import BFGS
from c2db.references import formation_energy
from c2db.prepare import convert_to_canonical_structure

Uvalues = {}

# From [acs comb sci 2.011, 13, 383-390, Setyawan et al.]
UTM = {'Ti': 4.4, 'V': 2.7, 'Cr': 3.5, 'Mn': 4.0, 'Fe': 4.6,
       'Co': 5.0, 'Ni': 5.1, 'Cu': 4.0, 'Zn': 7.5, 'Ga': 3.9,
       'Nb': 2.1, 'Mo': 2.4, 'Tc': 2.7, 'Ru': 3.0, 'Rh': 3.3,
       'Pd': 3.6, 'Cd': 2.1, 'In': 1.9,
       'Ta': 2.0, 'W': 2.2, 'Re': 2.4, 'Os': 2.6, 'Ir': 2.8, 'Pt': 3.0}

for key, value in UTM.items():
    Uvalues[key] = ':d,{},0'.format(value)


def relax_done(fname, emin=-np.inf):
    """Check if a relaxation is done"""
    if world.rank == 0:
        result = relax_done_master(fname, emin=emin)
    else:
        result = None
    return broadcast(result)


def relax_done_master(fname, fmax=0.01, smax=0.002, emin=-np.inf):
    if not Path(fname).is_file():
        return None, False

    try:
        slab = read(fname, parallel=False)
    except (IOError, UnknownFileTypeError):
        return None, False

    if slab.calc is None:
        return slab, False

    e = slab.get_potential_energy()
    f = slab.get_forces()
    s = slab.get_stress()

    done = e < emin or (f**2).sum(1).max() <= fmax**2 and abs(s).max() <= smax

    return slab, done


def relax(slab, tag, kptdens=6.0, width=0.05, emin=-np.inf):
    name = 'relax-' + tag

    kwargs = dict(txt=name + '.txt',
                  mode=PW(800),
                  xc='PBE',
                  basis='dzp',
                  kpts={'density': kptdens, 'gamma': True},
                  # This is the new default symmetry settings
                  symmetry={'do_not_symmetrize_the_density': True},
                  occupations=FermiDirac(width=width))

    if tag.endswith('+u'):
        # Try to get U values from previous image
        try:
            u = ulmopen('{}.traj'.format(name))
            setups = u[-1].calculator.get('parameters', {})['setups']
            u.close()
        except (FileNotFoundError, KeyError, InvalidULMFileError):
            # Read from standard values
            symbols = set(slab.get_chemical_symbols())
            setups = {symbol: Uvalues[symbol] for symbol in symbols
                      if symbol in Uvalues}
        kwargs['setups'] = setups
        world.barrier()

    slab.calc = GPAW(**kwargs)
    opt = BFGS(slab,
               logfile=name + '.log',
               trajectory=Trajectory(name + '.traj', 'a', slab))
    try:
        opt.run(fmax=0.01, smax=0.002, smask=[1, 1, 1, 1, 1, 1], emin=emin)
    except KohnShamConvergenceError:
        try:
            kwargs.update(kpts={'density': 9.0, 'gamma': True},
                          occupations=FermiDirac(width=0.02),
                          maxiter=999)
            slab.calc = GPAW(**kwargs)
            opt = BFGS(slab,
                       logfile=name + '.log',
                       trajectory=Trajectory(name + '.traj', 'a', slab))
            opt.run(fmax=0.01, smax=0.002, smask=[1, 1, 1, 1, 1, 1], emin=emin)
        except KohnShamConvergenceError:
            kwargs.update(occupations=FermiDirac(width=0.2))
            slab.calc = GPAW(**kwargs)
            opt = BFGS(slab,
                       logfile=name + '.log',
                       trajectory=Trajectory(name + '.traj', 'a', slab))
            opt.run(fmax=0.01, smax=0.002, smask=[1, 1, 1, 1, 1, 1], emin=emin)


def relax_all(prototype=None, U=False, states=None):
    """Relax atomic positions and unit cell.

    Different magnetic states will be tried: non-magnetic, ferro-magnetic, ...

    A gs.gpw file will be written for the most stable configuration.
    """

    info = readinfo()
    if states is None:
        states = info.get('states')
    nm = 'nm+u' if U else 'nm'
    fm = 'fm+u' if U else 'fm'
    afm = 'afm+u' if U else 'afm'
    if states is None:
        states = [nm, fm, afm]

    output = []
    for state in [nm, fm, afm]:
        output.append(relax_done('relax-{}.traj'.format(state)))

    slab1, done1 = output[0]
    slab2, done2 = output[1]
    slab3, done3 = output[2]

    hform1 = np.inf
    hform2 = np.inf
    hform3 = np.inf

    # Non-magnetic:
    if nm in states:
        if not done1:
            if slab1 is None:
                if U and Path('relax-nm.traj').exists():
                    try:
                        slab1 = read('relax-nm.traj')
                    except UnknownFileTypeError:
                        pass
                if slab1 is None:
                    fnames = list(Path('.').glob('start.*'))
                    assert len(fnames) == 1, fnames
                    slab1 = read(str(fnames[0]))
                    niggli_reduce(slab1)
            slab1.set_initial_magnetic_moments(None)
            try:
                relax(slab1, nm)
            except RuntimeError:
                # The atoms might be too close together
                # so enlarge unit cell
                slab1.set_cell(slab1.get_cell() * 2, scale_atoms=True)
                relax(slab1, nm)

        hform1 = formation_energy(slab1) / len(slab1)
        slab1.calc = None

    # Ferro-magnetic:
    if fm in states:
        if slab2 is None:
            slab2 = slab1.copy()
            slab2.set_initial_magnetic_moments([1] * len(slab2))

        if not done2:
            relax(slab2, fm)

        magmom = slab2.get_magnetic_moment()
        if abs(magmom) > 0.1:
            hform2 = formation_energy(slab2) / len(slab2)
            # Create subfolder early so that fm-tasks can begin:
            if world.rank == 0 and not Path(fm).is_dir():
                Path(fm).mkdir()
        slab2.calc = None

    # Antiferro-magnetic:
    if afm in states:
        if slab3 is None:
            if slab2 is None:
                # Special case.  Only afm relaxation is done
                fnames = list(Path('.').glob('start.*'))
                assert len(fnames) == 1, fnames
                slab3 = read(str(fnames[0]))
                niggli_reduce(slab3)
            else:
                slab3 = slab2.copy()
            magnetic = magnetic_atoms(slab3)
            nmag = sum(magnetic)
            if nmag == 2:
                magmoms = np.zeros(len(slab3))
                a1, a2 = np.where(magnetic)[0]
                magmoms[a1] = 1.0
                magmoms[a2] = -1.0
                slab3.set_initial_magnetic_moments(magmoms)
            else:
                done3 = True
                slab3 = None

        if not done3:
            relax(slab3, afm)

        if slab3 is not None:
            magmom = slab3.get_magnetic_moment()
            magmoms = slab3.get_magnetic_moments()
            if abs(magmom) < 0.02 and abs(magmoms).max() > 0.1:
                hform3 = formation_energy(slab3) / len(slab3)
            else:
                hform3 = np.inf
            slab3.calc = None

    hform = min(hform1, hform2, hform3)

    if hform1 > hform + 0.01:  # assume precison of 10 meV per atom
        hform1 = np.inf

    states = []
    for state, h, slab in [(nm, hform1, slab1),
                           (fm, hform2, slab2),
                           (afm, hform3, slab3)]:
        if h < np.inf:
            states.append(state)
            if world.rank == 0 and not Path(state).is_dir():
                Path(state).mkdir()


def get_parser():
    parser = argparse.ArgumentParser(description='Relax atomic structure')
    parser.add_argument('-p', '--prototype', help='One of MoS2, CdI2, ...')
    parser.add_argument('-U', '--plusu', help='Do +U calculation',
                        action='store_true')
    parser.add_argument('--states', help='list of nm, fm, afm', nargs='+',
                        default=None)
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args()
    prototype = args.prototype
    U = args.plusu
    states = args.states
    
    if prototype is None:
        prototype = readinfo().get('prototype')

    if prototype and not Path('info.json').is_file():
        with open('info.json', 'w') as fd:
            fd.write(json.dumps({'prototype': prototype}))

    relax_all(prototype=prototype, U=U, states=states)


if __name__ == '__main__':
    main()

