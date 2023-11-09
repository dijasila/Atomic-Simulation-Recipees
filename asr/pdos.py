"""Projected density of states."""
import numpy as np
from collections import defaultdict
from ase import Atoms
from asr.utils import magnetic_atoms
from asr.core import command, option, read_json, ASRResult
from asr.result.resultdata import PDResult, PdosResult


# Recipe tests:

params = "{'mode':{'ecut':200,...},'kpts':{'density':2.0},...}"
ctests = []
ctests.append({'description': 'Test the refined ground state of Si',
               'name': 'test_asr.pdos_Si_gpw',
               'tags': ['gitlab-ci'],
               'cli': ['asr run "setup.materials -s Si2"',
                       'ase convert materials.json structure.json',
                       'asr run "setup.params '
                       f'asr.gs@calculate:calculator {params} '
                       'asr.pdos@calculate:kptdensity 3.0 '
                       'asr.pdos@calculate:emptybands 5"',
                       'asr run gs',
                       'asr run pdos@calculate',
                       'asr run database.fromtree',
                       'asr run "database.browser --only-figures"']})

tests = []
tests.append({'description': 'Test the pdos of Si (cores=1)',
              'name': 'test_asr.pdos_Si_serial',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params '
                      f'asr.gs@calculate:calculator {params} '
                      'asr.pdos@calculate:kptdensity 3.0 '
                      'asr.pdos@calculate:emptybands 5"',
                      'asr run gs',
                      'asr run pdos',
                      'asr run database.fromtree',
                      'asr run "database.browser --only-figures"']})
tests.append({'description': 'Test the pdos of Si (cores=2)',
              'name': 'test_asr.pdos_Si_parallel',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params '
                      f'asr.gs@calculate:calculator {params} '
                      'asr.pdos@calculate:kptdensity 3.0 '
                      'asr.pdos@calculate:emptybands 5"',
                      'asr run gs',
                      'asr run -p 2 pdos',
                      'asr run database.fromtree',
                      'asr run "database.browser --only-figures"']})


# ---------- Main functionality ---------- #


# ----- Slow steps ----- #


@command(module='asr.pdos',
         creates=['pdos.gpw'],
         tests=ctests,
         requires=['gs.gpw'],
         dependencies=['asr.gs'])
@option('-k', '--kptdensity', type=float, help='K-point density')
@option('--emptybands', type=int, help='number of empty bands to include')
def calculate(kptdensity: float = 20.0, emptybands: int = 20) -> ASRResult:
    from asr.utils.refinegs import refinegs
    refinegs(selfc=False,
             kptdensity=kptdensity, emptybands=emptybands,
             gpw='pdos.gpw', txt='pdos.txt')


# ----- Fast steps ----- #


@command(module='asr.pdos',
         requires=['results-asr.gs.json', 'pdos.gpw'],
         tests=tests,
         dependencies=['asr.gs', 'asr.pdos@calculate'],
         returns=PDResult)
def main() -> PDResult:
    from gpaw import GPAW
    from ase.parallel import parprint
    from asr.magnetic_anisotropy import get_spin_axis

    # Get refined ground state with more k-points
    calc = GPAW('pdos.gpw')

    dos1 = calc.dos(shift_fermi_level=False)
    theta, phi = get_spin_axis()
    dos2 = calc.dos(soc=True, theta=theta, phi=phi, shift_fermi_level=False)

    results = {}

    # Calculate the dos at the Fermi energy
    parprint('\nComputing dos at Ef', flush=True)
    results['dos_at_ef_nosoc'] = dos1.raw_dos([dos1.fermi_level],
                                              width=0.0)[0]
    parprint('\nComputing dos at Ef with spin-orbit coupling', flush=True)
    results['dos_at_ef_soc'] = dos2.raw_dos([dos2.fermi_level],
                                            width=0.0)[0]

    # Calculate pdos
    parprint('\nComputing pdos', flush=True)
    results['pdos_nosoc'] = pdos(dos1, calc)
    parprint('\nComputing pdos with spin-orbit coupling', flush=True)
    results['pdos_soc'] = pdos(dos2, calc)

    return results


# ---------- Recipe methodology ---------- #


# ----- PDOS ----- #


def pdos(dos, calc):
    """Do a single pdos calculation.

    Main functionality to do a single pdos calculation.
    """
    from asr.core import singleprec_dict

    # Do calculation
    e_e, pdos_syl, symbols, ef = calculate_pdos(dos, calc)

    return PdosResult.fromdata(
        efermi=ef,
        symbols=symbols,
        energies=e_e,
        pdos_syl=singleprec_dict(pdos_syl))


def calculate_pdos(dos, calc):
    """Calculate the projected density of states.

    Returns
    -------
    energies : nd.array
        energies 10 eV under and above Fermi energy
    pdos_syl : defaultdict
        pdos for spin, symbol and orbital angular momentum
    symbols : list
        chemical symbols in Atoms object
    efermi : float
        Fermi energy

    """
    import gpaw.mpi as mpi
    from gpaw.utilities.progressbar import ProgressBar
    from ase.utils import DevNull

    zs = calc.atoms.get_atomic_numbers()
    chem_symbols = calc.atoms.get_chemical_symbols()
    efermi = calc.get_fermi_level()
    l_a = get_l_a(zs)

    ns = calc.get_number_of_spins()
    gaps = read_json('results-asr.gs.json').get('gaps_nosoc')
    e1 = gaps.get('vbm') or gaps.get('efermi')
    e2 = gaps.get('cbm') or gaps.get('efermi')
    e_e = np.linspace(e1 - 3, e2 + 3, 500)

    # We distinguish in (spin(s), chemical symbol(y), angular momentum (l)),
    # that is if there are multiple atoms in the unit cell of the same chemical
    # species, their pdos are added together.
    pdos_syl = defaultdict(float)
    s_i = [s for s in range(ns) for a in l_a for l in l_a[a]]
    a_i = [a for s in range(ns) for a in l_a for l in l_a[a]]
    l_i = [l for s in range(ns) for a in l_a for l in l_a[a]]
    sal_i = [(s, a, l) for (s, a, l) in zip(s_i, a_i, l_i)]

    # Set up progressbar
    if mpi.world.rank == 0:
        pb = ProgressBar()
    else:
        devnull = DevNull()
        pb = ProgressBar(devnull)

    for _, (spin, a, l) in pb.enumerate(sal_i):
        symbol = chem_symbols[a]

        p = dos.raw_pdos(e_e, a, 'spdfg'.index(l), None, spin, 0.0)

        # Store in dictionary
        key = ','.join([str(spin), str(symbol), str(l)])
        pdos_syl[key] += p

    return e_e, pdos_syl, calc.atoms.get_chemical_symbols(), efermi


def get_l_a(zs):
    """Define which atoms and angular momentum to project onto.

    Parameters
    ----------
    zs : [z1, z2, ...]-list or array
        list of atomic numbers (zi: int)

    Returns
    -------
    l_a : {int: str, ...}-dict
        keys are atomic indices and values are a string such as 'spd'
        that determines which angular momentum to project onto or a
        given atom

    """
    lantha = range(58, 72)
    acti = range(90, 104)

    zs = np.asarray(zs)
    l_a = {}
    atoms = Atoms(numbers=zs)
    mag_elements = magnetic_atoms(atoms)
    for a, (z, mag) in enumerate(zip(zs, mag_elements)):
        if z in lantha or z in acti:
            l_a[a] = 'spdf'
        else:
            l_a[a] = 'spd' if mag else 'sp'
    return l_a


if __name__ == '__main__':
    main.cli()
