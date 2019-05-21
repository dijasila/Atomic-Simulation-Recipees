from pathlib import Path
import numpy as np
from ase.io import read, write, Trajectory
from ase.io.formats import UnknownFileTypeError
from ase.io.ulm import open as ulmopen
from ase.io.ulm import InvalidULMFileError
from ase.parallel import world, broadcast

from asr.utils import command, option
from asr.utils.bfgs import BFGS

import click


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


def relax(atoms, name, kptdens=6.0, ecut=800, width=0.05, emin=-np.inf,
          smask=None, xc='PBE', plusu=False, dftd3=True):

    if dftd3:
        from ase.calculators.dftd3 import DFTD3

    trajname = f'{name}.traj'

    # Are we done?
    atoms_relaxed, done = relax_done(trajname)

    if atoms_relaxed is not None:
        atoms = atoms_relaxed

    if done:
        return atoms

    if smask is None:
        nd = int(np.sum(atoms.get_pbc()))
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
                  xc=xc,
                  basis='dzp',
                  kpts={'density': kptdens, 'gamma': True},
                  # This is the new default symmetry settings
                  symmetry={'do_not_symmetrize_the_density': True},
                  occupations={'name': 'fermi-dirac', 'width': width})

    if plusu:
        # Try to get U values from previous image
        try:
            u = ulmopen(f'{name}.traj')
            setups = u[-1].calculator.get('parameters', {})['setups']
            u.close()
        except (FileNotFoundError, KeyError, InvalidULMFileError):
            # Read from standard values
            symbols = set(atoms.get_chemical_symbols())
            setups = {symbol: Uvalues[symbol] for symbol in symbols
                      if symbol in Uvalues}
        kwargs['setups'] = setups
        world.barrier()

    from asr.utils.gpaw import GPAW, KohnShamConvergenceError
    dft = GPAW(**kwargs)
    if dftd3:
        calc = DFTD3(dft=dft)
    else:
        calc = dft
    atoms.calc = calc

    opt = BFGS(atoms,
               logfile=name + '.log',
               trajectory=Trajectory(name + '.traj', 'a', atoms))
    try:
        opt.run(fmax=0.01, smax=0.002, smask=smask, emin=emin)
    except KohnShamConvergenceError:
        try:
            kwargs.update(kpts={'density': 9.0, 'gamma': True},
                          occupations={'name': 'fermi-dirac', 'width': 0.02},
                          maxiter=999)
            dft = GPAW(**kwargs)
            if dftd3:
                calc = DFTD3(dft=dft)
            else:
                calc = dft
            atoms.calc = calc

            opt = BFGS(atoms,
                       logfile=name + '.log',
                       trajectory=Trajectory(name + '.traj', 'a', atoms))
            opt.run(fmax=0.01, smax=0.002, smask=smask, emin=emin)
        except KohnShamConvergenceError:
            kwargs.update(occupations={'name': 'fermi-dirac', 'width': 0.2})
            dft = GPAW(**kwargs)
            if dftd3:
                calc = DFTD3(dft=dft)
            else:
                calc = dft
            atoms.calc = calc
            opt = BFGS(atoms,
                       logfile=name + '.log',
                       trajectory=Trajectory(name + '.traj', 'a', atoms))
            opt.run(fmax=0.01, smax=0.002, smask=smask, emin=emin)

    return atoms


@command('asr.relax')
@option('--ecut', default=800,
        help='Energy cutoff in electronic structure calculation')
@option('--kptdens', default=6.0,
        help='Kpoint density')
@option('-U', '--plusu', help='Do +U calculation',
        is_flag=True)
@option('--xc', default='PBE', help='XC-functional')
@option('--d3/--nod3', default=True, help='Relax with vdW D3')
@click.pass_context
def main(ctx, plusu, ecut, kptdens, xc, d3):
    """Relax atomic positions and unit cell."""
    msg = ('You cannot have a structure.json file '
           'if you relax the structure because this is '
           'what the relax recipe produces. You should '
           'call your file unrelaxed.json!')
    assert not Path('structure.json').is_file(), msg
    atoms, done = relax_done('relax.traj')

    if not done:
        if atoms is None:
            if Path('relax.traj').exists():
                try:
                    atoms = read('relax.traj')
                except UnknownFileTypeError:
                    pass
            if atoms is None:
                atoms = read('unrelaxed.json')
        # Relax the structure
        relax(atoms, name='relax', ecut=ecut,
              kptdens=kptdens, xc=xc, plusu=plusu, dftd3=d3)

    toten = atoms.get_potential_energy()

    # Save to results-relax.json
    data = {'params': ctx.params,
            'toten': toten}
    from asr.utils import write_json
    write_json('results-relax.json', data)

    # Save atomic structure
    write('structure.json', atoms)


group = 'structure'
resources = '8:xeon8:10h'
creates = ['results-relax.json']

if __name__ == '__main__':
    main()
