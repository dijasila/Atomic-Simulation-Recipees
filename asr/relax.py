import json
from pathlib import Path
import numpy as np
from ase.io import read, write, Trajectory
from ase.io.formats import UnknownFileTypeError
from ase.io.ulm import open as ulmopen
from ase.io.ulm import InvalidULMFileError
from ase.parallel import world
from ase import Atoms

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


def is_relax_done(atoms, fmax=0.01, smax=0.002):
    f = atoms.get_forces()
    s = atoms.get_stress()
    done = (f**2).sum(1).max() <= fmax**2 and abs(s).max() <= smax

    return done


class SpgGroupAtoms(Atoms):

    @classmethod
    def from_atoms(cls, atoms):
        # Due to technicalities we cannot mess with the __init__ constructor
        # -> therefore we make our own
        return cls(atoms)

    def set_symmetries(self, symmetries, translations, atomsmap):
        self.op_scc = symmetries
        self.t_sc = translations
        self.op_svv = [np.linalg.inv(self.cell).dot(op_cc).dot(self.cell) for
                       op_cc in symmetries]
        self.nsym = len(symmetries)
        self.a_sa = atomsmap

    def get_stress(self, voigt=True, *args, **kwargs):
        sigma0_vv = Atoms.get_stress(self, voigt=False, *args, **kwargs)

        sigma_vv = np.zeros((3, 3))
        for op_vv in self.op_svv:
            sigma_vv += np.dot(np.dot(op_vv, sigma0_vv), op_vv.T)
        sigma_vv /= self.nsym

        if voigt:
            return sigma_vv.flat[[0, 4, 8, 5, 2, 1]]

        return sigma_vv

    def get_forces(self, *args, **kwargs):
        f0_av = Atoms.get_forces(self, *args, **kwargs)
        f_av = np.zeros_like(f0_av)
        for map_a, op_vv in zip(self.a_sa, self.op_svv):
            for a1, a2 in enumerate(map_a):
                f_av[a2] += np.dot(f0_av[a1], op_vv)
        f_av /= self.nsym
        return f_av


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

    from ase.calculators.calculator import kpts2sizeandoffsets
    size, _ = kpts2sizeandoffsets(density=kptdensity, atoms=atoms)
    kwargs = dict(txt=name + '.txt',
                  mode={'name': 'pw', 'ecut': ecut},
                  xc=xc,
                  basis='dzp',
                  symmetry={'symmorphic': False},
                  convergence={'forces': 1e-4},
                  kpts={'size': size, 'gamma': True},
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
    from gpaw.symmetry import atoms2symmetry
    symmetry = atoms2symmetry(atoms)
    atoms = SpgGroupAtoms.from_atoms(atoms)
    atoms.set_symmetries(symmetries=symmetry.op_scc,
                         translations=symmetry.ft_sc,
                         atomsmap=symmetry.a_sa)
    dft = get_calculator()(**kwargs)
    if dftd3:
        calc = DFTD3(dft=dft)
    else:
        calc = dft
    atoms.calc = calc

    from asr.setup.symmetrize import atomstospgcell as ats
    spgname, number = spglib.get_spacegroup(ats(read('unrelaxed.json')),
                                            symprec=1e-4,
                                            angle_tolerance=0.1).split()

    from ase.constraints import ExpCellFilter

    smask = [1, 1, 1, 1, 1, 1]
    filter = ExpCellFilter(atoms, mask=smask)
    from ase.optimize.bfgs import BFGS
    opt = BFGS(filter,
               logfile=name + '.log',
               trajectory=Trajectory(name + '.traj', 'a', atoms))

    # fmax=0 here because we have implemented our own convergence criteria
    runner = opt.irun(fmax=0)
    for _ in runner:
        # Check that the symmetry has not been broken
        spgname2, number2 = spglib.get_spacegroup(ats(atoms),
                                                  symprec=1e-4,
                                                  angle_tolerance=0.1).split()

        if not number == number2:
            # Log the last step
            opt.log()
            opt.call_observers()
            msg = ('The symmetry was broken during the relaxation! '
                   f'The initial spacegroup was {spgname} {number} '
                   f'but it changed to {spgname2} {number2} during '
                   'the relaxation.')
            raise AssertionError(msg)

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
    msg = ('You cannot already have a structure.json file '
           'when you relax a structure, because this is '
           'what the relax recipe is supposed to produce. You should '
           'name your original/start structure "unrelaxed.json!"')
    assert not Path('structure.json').is_file(), msg
    try:
        atoms = read('relax.traj')
    except (IOError, UnknownFileTypeError):
        atoms = read('unrelaxed.json')

    # Relax the structure
    atoms, calc, dft, kwargs = relax(atoms, name='relax', ecut=ecut,
                                     kptdensity=kptdensity, xc=xc,
                                     plusu=plusu, dftd3=d3, width=width)

    edft = dft.get_potential_energy(atoms)
    etot = atoms.get_potential_energy()

    # Save atomic structure
    write('structure.json', atoms)

    from asr.utils import write_json
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
