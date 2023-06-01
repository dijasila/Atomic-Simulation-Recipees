from asr.core import read_json, command, option, AtomsFile, DictStr, prepare_result, ASRResult
import numpy as np
from ase import Atoms
import os
from asr.utils.bilayerutils import translation
from ase.db import connect
import shutil
from asr.utils.symmetry import atoms2symmetry
from spglib import get_symmetry_dataset

c2db = connect('/home/niflheim2/cmr/C2DB-ASR/collected-databases/c2db-2021-06-24-extra.db')

def calc_setup_old(settings): 
 
    from gpaw import MixerDif, Mixer, MixerSum, PW, GPAW
    from ase.calculators.dftd3 import DFTD3

    calcsettings = {}
    d3 = settings.pop('d3', True)
    _ = settings.pop('mode', None)  # Mode not currently used. Here for back-comp

    mixersettings = settings.pop('mixer', None)
    if mixersettings == 'mixerdif':
        mixersettings = {'type': 'mixerdif'}
    elif mixersettings == 'mixer':
        mixersettings = {'type': 'mixersum'} 

    if type(mixersettings) != dict:
        mixersettings = {'type': 'default',
                         'beta': None, 'nmaxold': None,
                         'weight': None}

    mixertype = mixersettings.pop('type', 'default')
    if mixertype == 'mixer':
        calcsettings['mixer'] = Mixer(**mixersettings)
    elif mixertype == 'mixersum':
        calcsettings['mixer'] = MixerSum(**mixersettings)
    elif mixertype != 'default':
        calcsettings['mixer'] = MixerDif(**mixersettings)

    calcsettings['mode'] = PW(settings.pop('PWE'))
    calcsettings['symmetry'] = {'symmorphic': False}
    calcsettings['occupations'] = {'name': 'fermi-dirac',
                                   'width': 0.05}
    calcsettings['poissonsolver'] = {'dipolelayer': 'xy'}
    calcsettings['nbands'] = '200%'
    calcsettings['txt'] = 'zscan.txt'

    calcsettings.update(settings)

    # 3d TM atoms which need a Hubbard U correction
    #TM3d_atoms = {'V':3.1, 'Cr':3.5, 'Mn':3.8, 'Fe':4.0, 'Co':3.3, 'Ni':6.4, 'Cu':4.0}
    #atom_ucorr = set([atom.symbol for atom in atoms if atom.symbol in TM3d_atoms])
    #U_corrections_dct = {symbol: f':d, {TM3d_atoms[symbol]}' for symbol in atom_ucorr}
    #calcsettings.update(setups=U_corrections_dct)

    calc = GPAW(**calcsettings)
    
    if d3:
        calc = DFTD3(dft=calc, cutoff=60)
     
    return calc






def calc_setup(settings):
    from gpaw import MixerDif, Mixer, MixerSum, PW, GPAW
    from ase.calculators.dftd3 import DFTD3
    
    # I don't want to pop data from stetting and then update it because I want to reuse this function with original setting
    calcsettings = {}

    try:
       mixersettings = settings['mixer'].copy()
    except:
       mixersettings = None
    
    mixertype = mixersettings.pop('type', 'mixersum')
    if type(mixersettings) != dict:
        mixersettings = {'beta': 0.02, 'nmaxold': 5,
                         'weight': 50} 
        calcsettings['mixer'] = MixerSum(**mixersettings)
    elif mixertype=='mixersum':
        calcsettings['mixer'] = MixerSum(**mixersettings)
    elif mixertype=='mixer':
        calcsettings['mixer'] = Mixer(**mixersettings)
    elif mixertype=='default' or mixertype=='mixerdif':
        calcsettings['mixer'] = MixerDif(**mixersettings)  
    

    calcsettings['mode'] = PW(settings['PWE'])
    calcsettings['xc'] = settings['xc']
    calcsettings['maxiter'] = settings['maxiter']
    calcsettings['kpts'] = settings['kpts']
    calcsettings['convergence'] = settings['convergence']
    calcsettings['symmetry'] = {'symmorphic': False}
    calcsettings['occupations'] = {'name': 'fermi-dirac',
                                   'width': 0.05}
    calcsettings['poissonsolver'] = {'dipolelayer': 'xy'}
    calcsettings['nbands'] = '200%'
    calcsettings['txt'] = 'zscan.txt'

    # 3d TM atoms which need a Hubbard U correction
    #TM3d_atoms = {'V':3.1, 'Cr':3.5, 'Mn':3.8, 'Fe':4.0, 'Co':3.3, 'Ni':6.4, 'Cu':4.0}
    #atom_ucorr = set([atom.symbol for atom in atoms if atom.symbol in TM3d_atoms])
    #U_corrections_dct = {symbol: f':d, {TM3d_atoms[symbol]}' for symbol in atom_ucorr}
    #calcsettings.update(setups=U_corrections_dct)

    calc = GPAW(**calcsettings)

    # This calculation has to have d3 so I want an error if it is not in the setting
    d3 = settings['d3']
    if d3: calc = DFTD3(dft=calc, cutoff=60)

    return calc


# This function is used in the magnetic recipe as well merge them into one
def check_magmoms(bilayer, magmoms_FM):
    atoms = bilayer.copy()

    symmetry = atoms2symmetry(atoms, tolerance=0.01, angle_tolerance=0.1)
    eq_atoms =  symmetry.dataset['equivalent_atoms']

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


    # I am changing this deviation matrix because I don't want to compare very small numbers
    """
    deviation_matrix_fm = []
    for x in mag_atoms:
        deviation_matrix_fm.append([ (abs(magmoms_FM[x]) - abs(magmoms_FM[y]))/abs(magmoms_FM[x]) for y in mag_atoms])
    deviation_matrix_fm = np.array(deviation_matrix_fm)
    """

    deviation_matrix_fm = []
    for x in mag_atoms:
        deviation_matrix_row = []
        for y in mag_atoms:
            # I don't want to compare very small numbers
            if abs(magmoms_FM[x])>=0.05 or abs(magmoms_FM[y])>=0.05:
               deviation_matrix_row.append((abs(magmoms_FM[x]) - abs(magmoms_FM[y]))/abs(magmoms_FM[x]))
            else:
               deviation_matrix_row.append(0)
        deviation_matrix_fm.append(deviation_matrix_row)
    deviation_matrix_fm = np.array(deviation_matrix_fm)
   
    ###############################
    check_values_fm = []
    for m, x, w1 in zip(deviation_matrix_fm, mag_atoms_type, mag_eq_atoms):
        for n, y, w2 in zip(m, mag_atoms_type, mag_eq_atoms):
            if abs(n) > 0.1 and x == y and w1==w2:
                check_values_fm.append(n)

    if len(check_values_fm) == 0:
       return True  #FM state healthy
    else:
       return False #FM state wrong (ildefined)


def get_energy(base, top, h, t_c, calc0, memo, mirror):
    
    # First check if height h has already been calculated
    # memo = [(height1, energy1), (height2, energy2), ...]
    # t for t in memo if np.allclose(t[0], h) 
    try:
        h0, e0, mag0 = next(t for t in memo if abs(t[0]-h)<0.001)
        return e0
    except StopIteration:
        pass

    tx, ty = t_c[0], t_c[1]

    # tx, ty are kept fixed throughout relaxation
    # h (interlayer distance) is changed to find optimal
    # separation
    atoms = translation(tx, ty, h, top, base, mirror)

    atoms.calc = calc0

    e = atoms.get_potential_energy()   
 
    # We want to check the magnetic moment is not changing much if the material has +U or if the monolayer was magnetic
    if "magmom" in read_json("../structure-c2db.json")[1]:   
       monolayer_magmom = read_json("../structure-c2db.json")[1]["magmom"]
       #bilayer_magmom = atoms.get_magnetic_moment() #this does not work because the calculator is DFTD3 (relax recipe problem)
       bilayer_magmom = atoms.calc.dft.get_magnetic_moment()
       bilayer_magmoms = atoms.calc.dft.get_magnetic_moments()   
       assert check_magmoms(atoms, bilayer_magmoms), '>>> Final magnetic moments are not accepted'
       #if not check_magmoms(atoms, bilayer_magmoms):
       #   return None
 
    else:
       bilayer_magmom = 0
 
    callback_fn(memo, h, e, bilayer_magmom)

    return e


def initial_displacement(atoms, distance):
    maxz = np.max(atoms.positions[:, 2])
    minz = np.min(atoms.positions[:, 2])

    return distance + (maxz - minz)


def callback_fn(energy_curve, h, energy, totmag):
    from gpaw import mpi
    energy_curve.append((h, energy, totmag))
    if mpi.rank == 0:
       np.save('energy_curve_corrected.npy', np.array(energy_curve))


def zscan_interlayer(calc, settings, atoms, top_layer, energy_curve, t_c, h, step, energy_tol, distance_tol, mirror, maxiter=12):

    step0 = step
    eold = 10.0
    enew = eold
    h += 2*step
    icount = 0
    h_restart = True

    enew_list = []

    print('before', abs(step)>distance_tol , icount<maxiter , enew!=None , (icount==0 or np.abs(enew-eold)>energy_tol))
    while (abs(step)>distance_tol or icount==0 or np.abs(enew-eold)>energy_tol) and icount<maxiter and enew!=None:
       print('Interlayer distance: ', h, 'Restart mode: ', h_restart)

       icount += 1
       if enew>eold:
          step *= (-0.5)
       h -= step
       eold = enew

       try: 
           enew = get_energy(atoms.copy(), top_layer.copy(), h, t_c, calc, energy_curve, mirror)
       except:
           if h_restart:
              print("Restart zscan with larger interlayer distances.")
              shutil.copy('./zscan.txt','./zscan-before-restart.txt')    
              h += 0.5
              step = step0
              h_restart = False
              calc = calc_setup(settings)
              try:
                 enew = get_energy(atoms.copy(), top_layer.copy(), h, t_c, calc, energy_curve, mirror)
              except:
                 pass

       # If the last 5 steps did not converge we want to stop
       enew_list.append(enew)
       if len(enew_list)>=5:
          if len(set(enew_list[-5:-1]))==1:
             icount = 1000
       
       print('inside', abs(step)>distance_tol , icount<maxiter , enew!=None , (icount==0 or np.abs(enew-eold)>energy_tol))

    # Here I want to make sure at least one distance smaller than min is checked
    curve = np.array(energy_curve)
    assert np.size(curve)>1, 'No zscan step converged'

    if abs(h-min(curve[:,0]))<0.0001:
       if enew>eold:
          step *= (-0.5)
       h -= step
       eold = enew
       enew = get_energy(atoms.copy(), top_layer.copy(), h, t_c, calc, energy_curve, mirror)

    if enew is None:
       status = False
       reason = f'None energy - Magmom issue - {icount} Step'
    elif abs(step)>0.03:#distance_tol:
       status = False
       reason = f'Interlayer distance tolerenece not acheived - {icount} Step'
    else:
       status = True
       reason = f'Successful - {icount} Step'

    return status, reason, energy_curve


def optimize_interlayer(calc, settings, atoms, top_layer, energy_curve, t_c, d0, tol, mirror):
    import scipy.optimize as sciop 

    def energy_fn(h):
        return get_energy(atoms.copy(), top_layer.copy(),
                          h[0], t_c, calc, energy_curve, mirror)

    opt_result = sciop.minimize(energy_fn, x0=d0, method="Nelder-Mead", tol=tol)

    if not opt_result.success:
        status = False
        reason = 'Relaxation failed in optimizer'
    else:
        status = True
        reason = 'Relaxation succeeded in optimizer'

    return status, reason, energy_curve


@prepare_result
class RelaxBilayerResult(ASRResult):
    method: str
    heights: np.ndarray
    energies: np.ndarray
    total_magmoms: np.ndarray
    optimal_height: float
    interlayer_distance: float
    energy: float
    curvature: float
    FittingParams: np.ndarray

    key_descriptions = dict(
        method='Interlayer calculation method',
        heights='Interlayer distance calculated during optimization',
        energies='Energies calculated during optimization',
        total_magmoms='Total magnetic moment in each interlayer distance',
        optimal_height='Minimum energy height',
        interlayer_distance='Distance between highest atom in the bottom layer from lowest atom in the top layer',
        energy='Energy at optimal height',
        curvature='Curvature at optimal height',
        FittingParams='Parameters for second order fit')

@command('asr.zscan')
@option('-a', '--atoms', help='Base layer',
        type=AtomsFile(), default='../structure-c2db.json')
@option('-s', '--settings', help='Relaxation settings',
        type=DictStr())
@option('--tol', help='Convergence threshold',
        type=float)
@option('-d', '--distance', help='Initial Distance',
        type=float)
@option('-v', '--vacuum', help='Extra vacuum',
        type=float)
@option('--method', help='Interlayer calculation method',
        type=str, default='optimize')
@option('--restart/--norestart', help='Delete memo and start relaxation from scratch',
        is_flag=True, type=bool)
@option('--outputname', help='Name of output file', type=str)
def main(atoms: Atoms,
         settings=None,
         tol: float = 1e-2,
         distance: float = 5,
         vacuum: float = 6,
         method: str = 'optimize',
         restart: bool = False,
         outputname: str = 'structure.json') -> ASRResult:


    settings: dict = {'d3': True,
                      'xc': 'PBE',
                      'PWE': 700, # 'mode': {'name': 'pw', 'ecut': 800}
                      'maxiter': 600,
                      'kpts': {'density': 6.0, 'gamma': False}, #'kpts': {'density': 6.0, 'gamma': True},
                      'convergence': {'energy': 0.000001, 'bands': 'CBM+3.0'},
                      'mixer': {'type': 'mixersum', # used for the first trouble shooting of magntic systems
                                'beta': 0.02, #0.02,
                                'nmaxold': 5, #None,
                                'weight': 50}, #None},
                      **(settings or {})}

    from asr.core import read_json
    from ase.io import read
    from gpaw import mpi
    import scipy.optimize as sciop
    from asr.stack_bilayer import translation
    
    if restart:
        if mpi.rank == 0:
            if os.path.exists('energy_curve_corrected.npy'):
                os.remove('energy_curve_corrected.npy')
        mpi.world.barrier()

    top_layer = read('toplayer.json')
    if not np.allclose(top_layer.cell, atoms.cell):
        top_layer.cell = atoms.cell.copy()
        top_layer.center()
        top_layer.write("corrected.json")

    t_c = np.array(read_json('translation.json')['translation_vector']).astype(float)

    try: mirror = read_json('transformdata.json')['Bottom_layer_Mirror']
    except: mirror = False

    d0 = initial_displacement(atoms, distance)
    maxz = np.max(atoms.positions[:, 2])
    minz = np.min(atoms.positions[:, 2])
    w = maxz - minz
    atoms.cell[2, 2] += vacuum + w
    atoms.cell[2, 0:2] = 0.0
    top_layer.cell = atoms.cell

    # set the final magmoms of the monolayer for the initial values of the bilayer
    if "magmom" in read_json("../structure-c2db.json")[1]:
       magmoms = read_json("../structure-c2db.json")[1]["magmoms"]
       atoms.set_initial_magnetic_moments(magmoms)
       top_layer.set_initial_magnetic_moments(magmoms)

    # Build the initial structure and save it
    # The reason we make this structure is if we wanted to renew the calculator in each step we can. 
    start_structure = translation(t_c[0], t_c[1], d0, top_layer.copy(), atoms.copy(), mirror)
    start_structure.write("startstructure.json")

    settings_original = settings.copy()
    calc = calc_setup(settings.copy())

    # Commented because I want to start zscan from scratch
    if os.path.exists('energy_curve_corrected.npy'):
        energy_curve = np.load('energy_curve_corrected.npy', allow_pickle=True)
        energy_curve = [(d, e, mag) for d, e, mag in energy_curve]
    else:
        energy_curve = []

    bottom_layer = atoms.copy()
    if method == 'zscan':
       step = 0.079
       energy_tol = 0.000001
       distance_tol = tol # this is what we get as input of the recipe
       status, reason, energy_curve = zscan_interlayer(calc, settings, bottom_layer, top_layer, energy_curve, t_c, d0, step, energy_tol, distance_tol, mirror, maxiter=200)

    elif method == 'optimize':
       status, reason, energy_curve = optimize_interlayer(calc, settings, bottom_layer, top_layer, energy_curve, t_c, d0, tol, mirror)


    curve = np.array(energy_curve)
    # We do this because maybe the last calculation is not the min energy
    ee=curve[:, 1]
    hh=curve[:, 0]
    sorted_index = sorted(range(len(ee)), key=lambda k: ee[k])
    e_min = ee[sorted_index[0]]
    h_min = hh[sorted_index[0]]

    if mpi.rank == 0:
        np.save('energy_curve_corrected.npy', np.array(energy_curve))

    if not status:
        print(reason)
        raise ValueError(f'Relaxation failed - {reason}')

    final_atoms = translation(t_c[0], t_c[1], h_min, top_layer, atoms, mirror)
    final_atoms.write(outputname)
    P = bilayer_stiffness(curve)

    # interlayer distance 
    natoms = len(final_atoms)
    max_bot = max(final_atoms.positions[0:natoms//2,2])
    min_top = min(final_atoms.positions[natoms//2:natoms,2])    
    interlayer = min_top - max_bot

    results = {'method': method,
               'heights': curve[:, 0],
               'energies': curve[:, 1],
               'total_magmoms': curve[:, 2],
               'optimal_height': h_min,
               'interlayer_distance': interlayer,
               'energy': e_min,
               'curvature': P[2],
               'FittingParams': P}

    return RelaxBilayerResult.fromdata(**results)


def webpanel():
    """Return a webpanel showing the binding energy curve.

    Also show the fit made to determine the stiffness.
    """
    raise NotImplementedError()


def bilayer_stiffness(energy_curve):
    """Calculate bilayer stiffness.

    We define the bilayer stiffness as the curvature
    of the binding energy curve at the minimum.
    That is, we are calculating an effective spring
    constant.

    For now we include this property calculation here
    since we can get it almost for free.
    """
    # We do a second order fit using points within
    # 0.01 eV of minimum
    ds = energy_curve[:, 0]
    es = energy_curve[:, 1]
    mine = np.min(es)
    window = 0.01

    X = ds[np.abs(es - mine) < window]
    Y = es[np.abs(es - mine) < window]

    I = np.array([X**0, X**1, X**2]).T

    P, residuals, rank, singulars = np.linalg.lstsq(I, Y, rcond=None)

    return P


if __name__ == '__main__':
    main.cli()
