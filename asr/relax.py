from pathlib import Path
import numpy as np
from ase.io import read, write, Trajectory
from ase.io.formats import UnknownFileTypeError
from ase.io.ulm import open as ulmopen
from ase.io.ulm import InvalidULMFileError
from ase.parallel import world, broadcast

from asr.utils import get_dimensionality, magnetic_atoms
from asr.utils.bfgs import BFGS
from asr.convex_hull import get_hof
from asr.utils import update_defaults

import click
from functools import partial
import json

option = partial(click.option, show_default=True)


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


def relax(slab, tag, kptdens=6.0, ecut=800, width=0.05, emin=-np.inf,
          smask=None):
    name = f'relax-{tag}'
    trajname = f'{name}.traj'

    # Are we done?
    slab_relaxed, done = relax_done(trajname)

    if slab_relaxed is not None:
        slab = slab_relaxed

    if done:
        return slab

    if smask is None:
        nd = get_dimensionality()
        if nd == 3:
            smask = [1, 1, 1, 0, 0, 0]
        elif nd == 2:
            smask = [1, 1, 0, 0, 0, 0]
        else:
            # nd == 1
            msg = 'Relax recipe not implemented for 1D structures'
            raise NotImplementedError(msg)

    kwargs = dict(txt=name + '.txt',
                  mode={'name': 'pw', 'ecut': ecut},
                  xc='PBE',
                  basis='dzp',
                  kpts={'density': kptdens, 'gamma': True},
                  # This is the new default symmetry settings
                  symmetry={'do_not_symmetrize_the_density': True},
                  occupations={'name': 'fermi-dirac', 'width': width})

    if tag.endswith('+u'):
        # Try to get U values from previous image
        try:
            u = ulmopen(f'{name}.traj')
            setups = u[-1].calculator.get('parameters', {})['setups']
            u.close()
        except (FileNotFoundError, KeyError, InvalidULMFileError):
            # Read from standard values
            symbols = set(slab.get_chemical_symbols())
            setups = {symbol: Uvalues[symbol] for symbol in symbols
                      if symbol in Uvalues}
        kwargs['setups'] = setups
        world.barrier()

    from asr.utils.gpaw import GPAW, KohnShamConvergenceError
    slab.calc = GPAW(**kwargs)
    opt = BFGS(slab,
               logfile=name + '.log',
               trajectory=Trajectory(name + '.traj', 'a', slab))
    try:
        opt.run(fmax=0.01, smax=0.002, smask=smask, emin=emin)
    except KohnShamConvergenceError:
        try:
            kwargs.update(kpts={'density': 9.0, 'gamma': True},
                          occupations={'name': 'fermi-dirac', 'width': 0.02},
                          maxiter=999)
            slab.calc = GPAW(**kwargs)
            opt = BFGS(slab,
                       logfile=name + '.log',
                       trajectory=Trajectory(name + '.traj', 'a', slab))
            opt.run(fmax=0.01, smax=0.002, smask=smask, emin=emin)
        except KohnShamConvergenceError:
            kwargs.update(occupations={'name': 'fermi-dirac', 'width': 0.2})
            slab.calc = GPAW(**kwargs)
            opt = BFGS(slab,
                       logfile=name + '.log',
                       trajectory=Trajectory(name + '.traj', 'a', slab))
            opt.run(fmax=0.01, smax=0.002, smask=smask, emin=emin)

    return slab


@click.command()
@update_defaults('asr.relax')
@click.argument('states', nargs=-1)
@option('--ecut', default=800,
        help='Energy cutoff in electronic structure calculation')
@option('--kptdens', default=6.0,
        help='Kpoint density')
@option('-U', '--plusu', help='Do +U calculation',
        is_flag=True)
@option('--save-all-states',
        help='Save all states and not only the most stable state(s)',
        is_flag=True)
@option('--references',
        help='References database when calculating HOF')
def main(plusu, states, ecut, kptdens, save_all_states, references):
    """Relax atomic positions and unit cell.

    STATES: list of nm (non-magnetic), fm (ferro-magnetic), afm
    (anti-ferro-magnetic; only work with two magnetic
    atoms in unit cell)
    """
    U = plusu
    nm = 'nm+u' if U else 'nm'
    fm = 'fm+u' if U else 'fm'
    afm = 'afm+u' if U else 'afm'
    if not states:
        states = [nm, fm, afm]

    output = []
    for state in [nm, fm, afm]:
        output.append(relax_done('relax-{}.traj'.format(state)))

    slab1, done1 = output[0]
    slab2, done2 = output[1]
    slab3, done3 = output[2]

    toten_nm = np.nan
    toten_fm = np.nan
    toten_afm = np.nan

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
            slab1.set_initial_magnetic_moments(None)
            try:
                relax(slab1, nm, ecut=ecut, kptdens=kptdens)
            except RuntimeError:
                # The atoms might be too close together
                # so enlarge unit cell
                slab1.set_cell(slab1.get_cell() * 2, scale_atoms=True)
                relax(slab1, nm, ecut=ecut, kptdens=kptdens)

        toten_nm = slab1.get_potential_energy()

    # Ferro-magnetic:
    if fm in states:
        if slab2 is None:
            slab2 = slab1.copy()
            slab2.calc = None
            slab2.set_initial_magnetic_moments([1] * len(slab2))

        if not done2:
            relax(slab2, fm, ecut=ecut, kptdens=kptdens)

        magmom = slab2.get_magnetic_moment()
        if abs(magmom) > 0.1:
            # Create subfolder early so that fm-tasks can begin:
            if world.rank == 0 and not Path(fm).is_dir():
                Path(fm).mkdir()
        toten_fm = slab2.get_potential_energy()

    # Antiferro-magnetic:
    if afm in states:
        if slab3 is None:
            if slab2 is None:
                # Special case.  Only afm relaxation is done
                fnames = list(Path('.').glob('start.*'))
                assert len(fnames) == 1, fnames
                slab3 = read(str(fnames[0]))
            else:
                slab3 = slab2.copy()
                slab3.calc = None
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
            relax(slab3, afm, ecut=ecut, kptdens=kptdens)

        if slab3 is not None:
            magmom = slab3.get_magnetic_moment()
            magmoms = slab3.get_magnetic_moments()
            toten_afm = slab3.get_potential_energy()

    for state, slab in [(nm, slab1),
                        (fm, slab2),
                        (afm, slab3)]:
        if slab is None:
            continue
        if world.rank == 0 and not Path(state).is_dir():
            Path(state).mkdir()

        name = state + '/start.json'
        if not Path(name).is_file():
            # Write start.traj file to folder
            write(name, slab)

    # Save to results-relax.json
    data = {'toten_nm': toten_nm,
            'toten_fm': toten_fm,
            'toten_afm': toten_afm}
    Path('results-relax.json').write_text(json.dumps(data))


group = 'Structure'
resources = '8:xeon8:10h'
creates = ['results-relax.json']

if __name__ == '__main__':
    main(standalone_mode=False)
