from ase.io import read
from asr.core import read_json, command, option, DictStr, ASRResult, argument
from asr.utils.bilayerutils import translation
import numpy as np
from ase.calculators.dftd3 import DFTD3
from typing import List


def convert_mixer(mixer):
    from gpaw import MixerSum, MixerDif

    if 'type' not in mixer:
        raise ValueError('Key "type" is required')

    if mixer['type'] == 'MixerSum':
        beta = mixer['beta']
        nmaxold = mixer['nmaxold']
        weight = mixer['weight']
        return MixerSum(beta=beta, nmaxold=nmaxold, weight=weight)
    elif mixer['type'] == 'MixerDif':
        beta = mixer['beta']
        nmaxold = mixer['nmaxold']
        weight = mixer['weight']
        return MixerDif(beta=beta, nmaxold=nmaxold, weight=weight)
    else:
        raise ValueError(f'Unrecognized mixer type: {mixer["type"]}')


def get_bilayer(atoms, top_layer, magmoms, config=None):
    assert config is not None, 'Must provide config'
    # Extract data necessary to construct the bilayer from
    # the separate layers. This allows us to easily set
    # the magnetic moments in the separate layers.
    relax_data = read_json("results-asr.relax_bilayer.json")
    h = relax_data["optimal_height"]
    t_c = np.array(read_json('translation.json')[
                   'translation_vector']).astype(float)

    # Make local copies so we dont modify input
    atoms = atoms.copy()
    top = top_layer.copy()

    # Set the magnetic moments
    if config.lower() == 'fm':
        atoms.set_initial_magnetic_moments(magmoms)
        top.set_initial_magnetic_moments(magmoms)
    elif config.lower() == 'afm':
        atoms.set_initial_magnetic_moments(magmoms)
        top.set_initial_magnetic_moments(-magmoms)
    else:
        raise ValueError(f'Configuration not recognized: {config}')

    # Combine the two layers
    bilayer = translation(t_c[0], t_c[1], h, top, atoms)

    return bilayer


TM3d_atoms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']


@command(module='asr.bilayer_magnetism',
         requires=['results-asr.magstate.json'],
         resources='24:10h',
         dependencies=['asr.magstate',
                       'asr.relax_bilayer'])
@argument('uatoms', nargs=-1, type=List[str], help='List of atoms to add Hubbard U to')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
@option('-u', '--hubbardu', type=float, default=3, help="Hubbard U correction")  # U = 3
@option('-m', '--mixer', help='help', type=DictStr())
def main(uatoms: List[str] = TM3d_atoms,
         calculator: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'maxiter': 5000,
        'convergence': {'bands': 'CBM+3.0', "energy": 1e-6},
        'nbands': '200%'},
        hubbardu: float = 3,
        mixer: dict = None) -> ASRResult:
    """Calculate the energy difference between FM and AFM configurations.

    Returns the energy difference between the FM and
    AFM configurations for a bilayer, where FM is
    defined as all spins pointing in the same directions,
    and AFM is defined as spins in the two layers
    pointing in opposite directions.

    If U is non-zero, Hubbard-U correction is used for 3d TM
    atoms, if there are any.

    If mixer is not None, a custom mixer is used for the GPAW calculation.
    """
    if mixer is not None:
        raise NotImplementedError  # "params" is not defined
        # params['mixer'] = convert_mixer(mixer)

    u = hubbardu
    atoms = read('../structure.json')
    top_layer = read('toplayer.json')

    # Fix the cell of the bottom and top layers
    # to be the bilayer cell
    cell = read("structure.json").cell
    atoms.cell = cell
    top_layer.cell = cell

    # Read the magnetic moments from the monolayers.
    # This is used as the starting point for the bilayer
    # magnetic moments.
    magmoms = read_json("../structure.json")[1]["magmoms"]

    atom_ucorr = set(atoms.symbol) & set(uatoms)
    U_corrections_dct = {symbol: f':d, {u}' for symbol in atom_ucorr}

    calculator.update(setups=U_corrections_dct)

    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    calc_fm = get_calculator_class(name)(**calculator, txt=f"fm_U{u}.txt")
    calc_afm = get_calculator_class(name)(**calculator, txt=f"afm_U{u}.txt")

    # We use cutoff=60 for the vdW correction to be consistent with
    # asr.relax_bilayer
    calc_fm_D3 = DFTD3(dft=calc_fm, cutoff=60)

    # FM Calculation
    bilayer_FM = get_bilayer(atoms, top_layer, magmoms, config="FM")
    initial_magmoms_FM = bilayer_FM.get_initial_magnetic_moments()

    bilayer_FM.set_calculator(calc_fm)
    final_magmoms_FM = bilayer_FM.get_magnetic_moments()

    bilayer_FM.set_calculator(calc_fm_D3)
    eFM = bilayer_FM.get_potential_energy()

    FM_syms = bilayer_FM.get_chemical_symbols()
    for x in [i for i, e in enumerate(FM_syms) if e in uatoms]:
        assert np.sign(final_magmoms_FM[x]) == np.sign(initial_magmoms_FM[x])

    # AFM Calculation
    calc_afm_D3 = DFTD3(dft=calc_afm, cutoff=60)

    bilayer_AFM = get_bilayer(atoms, top_layer, magmoms, config="AFM")
    bilayer_AFM.set_calculator(calc_afm_D3)

    eAFM = bilayer_AFM.get_potential_energy()

    eDIFF = eFM - eAFM

    M_AFM = calc_afm.get_magnetic_moment()
    if eDIFF > 0 or not np.allclose(np.linalg.norm(M_AFM), 0):
        calc_afm.write(f'gs_U{u}.gpw')
    else:
        calc_fm.write(f'gs_U{u}.gpw')

    return dict(eFM=eFM, eAFM=eAFM,
                eDIFF=eDIFF,
                AFMNormIsZero=np.allclose(np.linalg.norm(M_AFM), 0))


if __name__ == '__main__':
    main.cli()
