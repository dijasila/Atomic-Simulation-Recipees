import json
from pathlib import Path
import numpy as np
from ase.io import read, write, Trajectory
from ase.io.formats import UnknownFileTypeError
from ase.io.ulm import open as ulmopen
from ase.io.ulm import InvalidULMFileError
from ase.parallel import world

from asr.utils import command, option
from gpaw import KohnShamConvergenceError

Uvalues = {}

# From [acs comb sci 2.011, 13, 383-390, Setyawan et al.]
UTM = {'Ti': 4.4, 'V': 2.7, 'Cr': 3.5, 'Mn': 4.0, 'Fe': 4.6,
       'Co': 5.0, 'Ni': 5.1, 'Cu': 4.0, 'Zn': 7.5, 'Ga': 3.9,
       'Nb': 2.1, 'Mo': 2.4, 'Tc': 2.7, 'Ru': 3.0, 'Rh': 3.3,
       'Pd': 3.6, 'Cd': 2.1, 'In': 1.9,
       'Ta': 2.0, 'W': 2.2, 'Re': 2.4, 'Os': 2.6, 'Ir': 2.8, 'Pt': 3.0}

for key, value in UTM.items():
    Uvalues[key] = ':d,{},0'.format(value)


def get_atoms(fname):
    try:
        atoms = read(fname)
    except (IOError, UnknownFileTypeError):
        return read('unrelaxed.json')

    return atoms


def is_relax_done(atoms, fmax=0.01, smax=0.002):
    f = atoms.get_forces()
    s = atoms.get_stress()
    done = (f**2).sum(1).max() <= fmax**2 and abs(s).max() <= smax

    return done


def relax(atoms, name, kptdensity=6.0, ecut=800, width=0.05, emin=-np.inf,
          smask=None, xc='PBE', plusu=False, dftd3=True):
    import spglib

    if dftd3:
        from ase.calculators.dftd3 import DFTD3

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
                  kpts={'density': kptdensity, 'gamma': True},
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

    from asr.calculators import get_calculator
    dft = get_calculator()(**kwargs)
    if dftd3:
        calc = DFTD3(dft=dft)
    else:
        calc = dft
    atoms.calc = calc

    spgname, number = spglib.get_spacegroup(read('unrelaxed.json'),
                                            symprec=1e-4).split()

    from ase.constraints import ExpCellFilter
    filter = ExpCellFilter(atoms, mask=smask)

    from ase.optimize import BFGS
    opt = BFGS(filter,
               logfile=name + '.log',
               trajectory=Trajectory(name + '.traj', 'a', atoms))

    # fmax=0 here because we have implemented our own convergence criteria
    runner = opt.irun(fmax=0)
    for _ in runner:
        # Check that the symmetry is has not been broken
        spgname2, number2 = spglib.get_spacegroup(atoms,
                                                  symprec=1e-4).split()

        assert number == number2, \
            ('The symmetry was broken during the relaxation! '
             f'The initial spacegroup was {spgname} {number} '
             f'but it changed to {spgname2} {number2} during '
             'the relaxation.')

        if is_relax_done(atoms, fmax=0.01, smax=0.002):
            break
        
    return atoms, calc, dft, kwargs


# Please note these are relative numbers that
# are multiplied on the original ones
known_exceptions = {KohnShamConvergenceError: {'kptdensity': 1.5,
                                               'width': 0.5}}


@command('asr.relax',
         known_exceptions=known_exceptions)
@option('--ecut', default=800,
        help='Energy cutoff in electronic structure calculation')
@option('--kptdensity', default=6.0,
        help='Kpoint density')
@option('-U', '--plusu', help='Do +U calculation',
        is_flag=True)
@option('--xc', default='PBE', help='XC-functional')
@option('--d3/--nod3', default=True, help='Relax with vdW D3')
@option('--width', default=0.05,
        help='Fermi-Dirac smearing temperature')
def main(plusu, ecut, kptdensity, xc, d3, width):
    """Relax atomic positions and unit cell.

    By default, this recipe takes the atomic structure in 'unrelaxed.json'
    and relaxes the structure including the DFTD3 van der Waals
    correction. The relaxed structure is saved to `structure.json` which can be
    processed by other recipes.

    \b
    Examples:
    Relax without using DFTD3
        asr run relax --nod3
    Relax using the LDA exchange-correlation functional
        asr run relax --xc LDA
    """
    msg = ('You cannot have a structure.json file '
           'if you relax the structure because this is '
           'what the relax recipe produces. You should '
           'call your original/start file "unrelaxed.json!"')
    assert not Path('structure.json').is_file(), msg
    atoms = get_atoms('relax.traj')

    # Relax the structure
    atoms, calc, dft, kwargs = relax(atoms, name='relax', ecut=ecut,
                                     kptdensity=kptdensity, xc=xc,
                                     plusu=plusu, dftd3=d3, width=width)

    edft = dft.get_potential_energy(atoms)
    etot = atoms.get_potential_energy()

    # Save atomic structure
    write('structure.json', atoms)

    from asr.utils import write_json
    kwargs.pop('txt')
    write_json('gs_params.json', kwargs)

    # Get setup fingerprints
    fingerprint = {}
    for setup in dft.setups:
        fingerprint[setup.symbol] = setup.fingerprint

    # Save to results_relax.json
    structure = json.loads(Path('structure.json').read_text())
    results = {'etot': etot,
               'edft': edft,
               'relaxedstructure': structure,
               '__key_descriptions__':
               {'etot': 'Total energy [eV]',
                'edft': 'DFT total energy [eV]',
                'relaxedstructure': 'Relaxed atomic structure'},
               '__setup_fingerprints__': fingerprint}
    return results


group = 'structure'
resources = '24:10h'
creates = ['results_relax.json']

if __name__ == '__main__':
    main()
