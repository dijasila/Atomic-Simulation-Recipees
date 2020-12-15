from ase.io import read
from asr.core import read_json, command, option, DictStr, argument
from asr.stack_bilayer import translation
from gpaw import GPAW, PW, FermiDirac
import numpy as np
from ase.calculators.dftd3 import DFTD3


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

@command(module='asr.bilayer_magnetism',
         requires=['results-asr.magstate.json'],
         resources='24:10h',
         dependencies=['asr.magstate',
                       'asr.relax_bilayer'])
@option('-m', '--mixer', help='help', type=DictStr())
@argument('u', type=float)  # U = 3
def main(u, mixer=None):
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

    # 3d TM atoms which need a Hubbard U correction
    TM3d_atoms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

    atom_ucorr = set([atom.symbol for atom in atoms
                      if atom.symbol in TM3d_atoms])
    U_corrections_dct = {symbol: f':d, {u}' for symbol in atom_ucorr}

    params = dict(mode=PW(800),
                  xc="PBE",
                  basis="dzp",
                  kpts={"density": 12.0, "gamma": True},
                  occupations=FermiDirac(0.05),
                  setups=U_corrections_dct,
                  convergence={"bands": "CBM+3.0"},
                  nbands="200%")

    if mixer is not None:
        params['mixer'] = convert_mixer(mixer)

    calc = GPAW(**params, txt=f"fm_U{u}.txt")
    # We use cutoff=60 for the vdW correction to be consistent with
    # asr.relax_bilayer
    calc = DFTD3(dft=calc, cutoff=60)

    # FM Calculation
    bilayer_FM = get_bilayer(atoms, top_layer, magmoms, config="FM")

    bilayer_FM.set_calculator(calc)
    eFM = bilayer_FM.get_potential_energy()

    # AFM Calculation
    calc_afm = GPAW(**params, txt=f"afm_U{u}.txt")
    calc_afm = DFTD3(dft=calc_afm, cutoff=60)

    bilayer_AFM = get_bilayer(atoms, top_layer, magmoms, config="AFM")
    bilayer_AFM.set_calculator(calc_afm)

    eAFM = bilayer_AFM.get_potential_energy()

    eDIFF = eFM - eAFM

    return dict(eFM=eFM, eAFM=eAFM, eDIFF=eDIFF)


if __name__ == '__main__':
    main.cli()
