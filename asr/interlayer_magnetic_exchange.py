from ase.io import read, write
from asr.core import read_json, command, option, DictStr, ASRResult
from asr.utils.bilayerutils import translation
import numpy as np
from ase.calculators.dftd3 import DFTD3
import os
from asr.utils.symmetry import atoms2symmetry
from spglib import get_symmetry_dataset

def check_magmoms(bilayer, magmoms_FM, magmoms_AFM, magmom_AFM, eq_atoms, dipz=0, dipz_threshold=1e-3):
    atoms = bilayer.copy()
    
    magnetic_atoms = []
    magnetic_atom_types = []
    mag_elements = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl']

    for atom in atoms:
        if atom.symbol in mag_elements:
            magnetic_atoms.append(1)
            magnetic_atom_types.append(atom.symbol)
        else:
            magnetic_atoms.append(0)
            magnetic_atom_types.append("")
                
    #mag_atoms = [x for x, z in enumerate(magnetic_atoms) if z == 1]
    mag_atoms = []
    mag_atoms_type = []
    mag_eq_atoms = []
    for iatom, z in enumerate(magnetic_atoms):
        if z==1: 
           mag_atoms.append(iatom)
           mag_atoms_type.append(magnetic_atom_types[iatom])
           mag_eq_atoms.append(eq_atoms[iatom])

    lines = []
    magmoms = []
        
    #deviation_matrix_fm = []
    #for x in mag_atoms:
    #    deviation_matrix_fm.append([ (abs(magmoms_FM[x]) - abs(magmoms_FM[y]))/abs(magmoms_FM[x]) for y in mag_atoms])
    #deviation_matrix_fm = np.array(deviation_matrix_fm)

    #deviation_matrix_afm = []
    #for x in mag_atoms:
    #    deviation_matrix_afm.append([ (abs(magmoms_AFM[x]) - abs(magmoms_AFM[y]))/abs(magmoms_AFM[x]) for y in mag_atoms])
    #deviation_matrix_afm = np.array(deviation_matrix_afm)

    # I am changing this because I don't want to compare very small numbers
    deviation_matrix_fm = []
    for x in mag_atoms:
        deviation_matrix_row = []
        for y in mag_atoms:
            if abs(magmoms_FM[x])>=0.05 or abs(magmoms_FM[y])>=0.05:
               deviation_matrix_row.append((abs(magmoms_FM[x]) - abs(magmoms_FM[y]))/abs(magmoms_FM[x]))
            else:
               deviation_matrix_row.append(0)
        deviation_matrix_fm.append(deviation_matrix_row)
    deviation_matrix_fm = np.array(deviation_matrix_fm)
    

    deviation_matrix_afm = []
    for x in mag_atoms:
        deviation_matrix_row = []
        for y in mag_atoms:
            if abs(magmoms_AFM[x])>=0.05 or abs(magmoms_AFM[y])>=0.05:
               deviation_matrix_row.append((abs(magmoms_AFM[x]) - abs(magmoms_AFM[y]))/abs(magmoms_AFM[x]))
            else:
               deviation_matrix_row.append(0)
        deviation_matrix_afm.append(deviation_matrix_row)
    deviation_matrix_afm = np.array(deviation_matrix_afm)


    ###############################
    check_values_fm = []
    for m, x, w1 in zip(deviation_matrix_fm, mag_atoms_type, mag_eq_atoms):
        for n, y, w2 in zip(m, mag_atoms_type, mag_eq_atoms):
            if abs(n) > 0.1 and x == y and w1==w2:
                check_values_fm.append(n)

    FM_state = 'FM' if (len(check_values_fm) == 0) else 'ildefined'

    ##############################
    check_values_afm = []
    if not deviation_matrix_afm == 'ildefined':
       for m, x, w1 in zip(deviation_matrix_afm, mag_atoms_type, mag_eq_atoms):
           for n, y, w2 in zip(m, mag_atoms_type, mag_eq_atoms):
               if abs(n) > 0.1 and x == y and w1==w2:
                   check_values_afm.append(n)

    if dipz<dipz_threshold and not np.allclose((magmom_AFM/len(mag_atoms)),0, atol=0.01):
        deviation_matrix_afm = 'ildefined'
        check_values_afm.append('ildefined')

    AFM_state = 'AFM' if (len(check_values_afm) == 0) else 'ildefined'    

    return FM_state, AFM_state



def get_bilayer(bilayer_atoms, magmoms, config=None):
    assert config is not None, 'Must provide config'
    bilayer = bilayer_atoms.copy()

    # Set the magnetic moments
    if config.lower() == 'fm':
        magmoms_bilayer = list(magmoms) + list(magmoms)
        bilayer.set_initial_magnetic_moments(magmoms_bilayer)
    
    elif config.lower() == 'afm':
        magmoms_bilayer = list(magmoms) + list(-magmoms)
        bilayer.set_initial_magnetic_moments(magmoms_bilayer)
    else:
        raise ValueError(f'Configuration not recognized: {config}')

    return bilayer


#@command(module='asr.interlayer_magnetic_exchange',
#         resources='24:10h',
#         dependencies=['asr.zscan'])

@command(module='asr.interlayer_magnetic_exchange',
         resources='24:10h')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
@option('-u', '--hubbardu', type=float, default=None, help="Hubbard U correction")
@option('-d', '--interlayer', type=float, default=None, help="Interlayer distance")
@option('-ml', '--mlfolder', type=str, help="monolayer folder")
@option('-mix', '--mixertype', type=str, default="sum", help="mixer: sum or difference")
def main(calculator: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'maxiter': 5000,
        'mixer': None,
        'poissonsolver' : {'dipolelayer': 'xy'},
        #'convergence': {'bands': 'CBM+3.0', "energy": 1e-6, "density": 1e-6},
        'convergence': None,
        'nbands': '200%'},
        mixertype: str = "sum",
        interlayer: float = -100.0, #if negative read from zscan
        mlfolder: str = '..',
        hubbardu: float = 0.0) -> ASRResult:

    if calculator['mixer'] is None:
        mixersetup = {"method": mixertype,
              "beta": 0.01,
              "history": 5,
              "weight": 50}
        calculator = {**calculator, 'mixer': mixersetup}

    convergence_energy = {'bands': 'CBM+3.0', "density": 1e-6, 'energy': {"name": "energy", "tol": 1e-6, "relative": False,  "n_old": 3} } 
    calculator = {**calculator, 'convergence': convergence_energy}

    """Calculate the energy difference between FM and AFM configurations.

    Returns the energy difference between the FM and
    AFM configurturations for a bilayer, where FM is
    defined as all spins pointing in the same directions,
    and AFM is defined as spins in the two layers
    pointing in opposite directions.

    If U is non-zero, Hubbard-U correction is used for 3d TM
    atoms, if there are any.

    """

    atoms = read(f'{mlfolder}/structure.json')
    bilayer_atoms = read('structure.json')

    u = hubbardu
    if hubbardu < 0:
        # 3d TM atoms which need a Hubbard U correction
        TM3d_atoms = {'V':3.1, 'Cr':3.5, 'Mn':3.8, 'Fe':4.0, 'Co':3.3, 'Ni':6.4, 'Cu':4.0}
        atom_ucorr = set([atom.symbol for atom in atoms if atom.symbol in TM3d_atoms])
        U_corrections_dct = {symbol: f':d, {TM3d_atoms[symbol]}' for symbol in atom_ucorr}
    elif hubbardu > 0:
        u = hubbardu
        TM3d_atoms = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
        atom_ucorr = set([atom.symbol for atom in atoms
                      if atom.symbol in TM3d_atoms])
        U_corrections_dct = {symbol: f':d, {u}' for symbol in atom_ucorr}
  
    if abs(hubbardu)>0.001:
       calculator.update(setups=U_corrections_dct)

    mag_elements = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl']


    # Read the magnetic moments from the monolayers.
    # This is used as the starting point for the bilayer magnetic moments.
    magmoms = read_json(f"{mlfolder}/structure.json")[1]["magmoms"]
                
    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    calc_fm = get_calculator_class(name)(**calculator, txt=f"fm_U.txt") #had {u} before
    calc_afm = get_calculator_class(name)(**calculator, txt=f"afm_U.txt") #had {u} before

    # We use cutoff=60 for the vdW correction to be consistent with
    calc_fm_D3 = DFTD3(dft=calc_fm, cutoff=60)

    # FM Calculation
    bilayer_FM = get_bilayer(bilayer_atoms, magmoms, config="FM")
    
    initial_magmoms_FM = bilayer_FM.get_initial_magnetic_moments()

    bilayer_FM.set_calculator(calc_fm)
    final_magmoms_FM = bilayer_FM.get_magnetic_moments()

    bilayer_FM.set_calculator(calc_fm_D3)
    eFM = bilayer_FM.get_potential_energy()
    bilayer_FM.get_forces()

    FM_syms = bilayer_FM.get_chemical_symbols()
    for x in [i for i, e in enumerate(FM_syms) if e in mag_elements]:
        mag_max = max(abs(np.array(initial_magmoms_FM))) 
        if abs(initial_magmoms_FM[x])>(0.1*mag_max):
           assert np.sign(final_magmoms_FM[x]) == np.sign(initial_magmoms_FM[x])

    # AFM Calculation
    calc_afm_D3 = DFTD3(dft=calc_afm, cutoff=60)

    bilayer_AFM = get_bilayer(bilayer_atoms, magmoms, config="AFM")
    bilayer_AFM.set_calculator(calc_afm_D3)

    eAFM = bilayer_AFM.get_potential_energy()
    bilayer_AFM.get_forces()

    eDIFF = eFM - eAFM

    magmoms_fm = calc_fm.get_magnetic_moments()
    M_FM = calc_fm.get_magnetic_moment()

    magmoms_afm = calc_afm.get_magnetic_moments()
    M_AFM = calc_afm.get_magnetic_moment()

    monolayer_dipole = read_json(f'{mlfolder}/results-asr.gs.json')['dipz']
    #eq_atoms = read_json('results-asr.structureinfo.json')['spglib_dataset']['equivalent_atoms']
    # I want to use less strict symmetry and also not depend on structureinfo
    symmetry = atoms2symmetry(bilayer_FM, tolerance=0.01, angle_tolerance=0.1)
    eq_atoms =  symmetry.dataset['equivalent_atoms']
    print(eq_atoms)

    FM_state, AFM_state = check_magmoms(atoms, magmoms_fm, magmoms_afm, M_AFM, eq_atoms, dipz=abs(monolayer_dipole), dipz_threshold=1e-3)

    #if abs(monolayer_dipole)<1e-3:
    #   print('Magmoms checked not to change much')
    #   eq_atoms = read_json('results-asr.structureinfo.json')['spglib_dataset']['equivalent_atoms']
    #   FM_state, AFM_state = check_magmoms(atoms, magmoms_fm, magmoms_afm, M_AFM, eq_atoms)
    #else:
    #   print('Magmoms not compared due to out-of-plane dipole')
    #   FM_state = 'FM'
    #   AFM_state = 'AFM'


    if FM_state == 'FM' and AFM_state == 'AFM':
        if eDIFF > 0:
            calc_afm.write(f'gs_U.gpw') #had {u} before
            bilayer_AFM.write(f'structure.json')
        else:
            calc_fm.write(f'gs_U.gpw') #had {u} before
            bilayer_FM.write(f'structure.json')

    if FM_state == 'FM' and AFM_state == 'ildefined':
        calc_fm.write(f'gs_U.gpw')  #had {u} before
        bilayer_FM.write(f'structure.json')
    
    if FM_state == 'ildefined' and AFM_state == 'AFM':
        calc_afm.write(f'gs_U.gpw')  #had {u} before
        bilayer_AFM.write(f'structure.json')


    return dict(eFM=eFM, eAFM=eAFM, eDIFF=eDIFF,
                M_AFM=M_AFM, M_FM=M_FM,
                magmoms_fm=magmoms_fm, magmoms_afm=magmoms_afm,
                FM_state=FM_state, AFM_state=AFM_state)

if __name__ == '__main__':
    main.cli()
