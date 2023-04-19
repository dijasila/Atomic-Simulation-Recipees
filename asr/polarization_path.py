import numpy as np
import os

from ase.io import read
from ase.units import _e

from gpaw.mpi import rank
from gpaw.berryphase import get_polarization_phase
from gpaw import GPAW, PW, Mixer, MixerDif, MixerSum

from asr.core import command, option, ASRResult, prepare_result

def remove_files(x):
    x = x
    if os.path.isfile(f'structure_{x}.gpw') is True:
        if rank == 0:
            os.remove(f'structure_{x}.gpw')
    if os.path.isfile(f'structure_{x}-berryphases.json') is True:
        if rank == 0:
            os.remove(f'structure_{x}-berryphases.json') 

@command(module='asr.polarization_path',
         requires=['results-asr.relax.json'],
         resources='40:2d')
@option('-p', '--path_points', help='help', type=int)
@option('-d', '--dimension', help='help', type=int)
@option('-r', '--replicas', help='help', type=int) 
@option('-ref', '--reference', help='help', type=int) 
def main(path_points = 15, dimension = 2, replicas = 4, reference=0) -> ASRResult:
    """
    Creates a path and computes the formal 
    polarization for each point along the path.

    Returns the spontaneous polarization,
    which is defined as the total change in
    the formal polarization as it goes from
    the non-polar to the polar structure.

    Path_points specifies how many points are 
    used the sampled the path. The default is
    15 points. 

    The dimension variable is used to determine 
    wheter the 2D area or the 3D volume should
    be used to compute the polarization. The 
    default is 2. 
    
    A hubbard U correction can be added if neccesary.
    This is typically relevant to insure that the 
    bandstructures are gapped in the initial and final 
    states as well as any state in between. An example
    would be antiferromagnetic materials.
    """

    path_points_array = np.linspace(0, 1, path_points)

    atoms_polar = read('structure.json')
    if reference == 0:
        atoms_non_polar = read('centrosymmetric_structure/structure.json')
    if reference == -1:
        atoms_non_polar = read('switched_structure/structure.json')

    p1 = atoms_non_polar.get_positions()
    p2 = atoms_polar.get_positions()
    p3 = p2 - p1

    if dimension == 2:
        cell_vc = (atoms_polar.get_cell().T) * 1e-10
        V = np.linalg.norm(np.cross(cell_vc[0], cell_vc[1]))
    elif dimension == 3:
        cell_vc = (atoms_polar.get_cell().T) * 1e-10
        V = (atoms_polar.get_volume()) * 1.0e-30 
    else:
        raise Exception(f'The dimension should be either 2 or 3. The value given was: {d}') 

    ## These parameters are the same, as those used in the formalpolarization recipe. 
    params = dict(mode=PW(800),
                      kpts={'density': 12, 'gamma': True},
                      symmetry='off',
                      mixer=MixerSum(0.02, 3, 100),
                      maxiter=5000,
                      occupations = {'name': 'fermi-dirac','width': 0.05},
                      convergence = {'eigenstates': 1e-11, 'density': 1e-7},
                      xc = 'PBE')
    
    Pa_path = np.ndarray(path_points, dtype=float)
    Pb_path = np.ndarray(path_points, dtype=float)
    Pc_path = np.ndarray(path_points, dtype=float)
    E = np.ndarray(path_points, dtype=float)

    n = replicas

    for i in np.arange(0, path_points):
        x = path_points_array[i]
        calc = GPAW(**params, txt=f"structure_{x}.txt")
  
        p4 = p1 + p3*x  ## Positions for structure x 
        atoms_non_polar.set_positions(p4)
        atoms_non_polar.set_calculator(calc)

        E[i] = atoms_non_polar.get_potential_energy()

        calc.write(f'structure_{x}.gpw', mode='all')
        
        phi_c = get_polarization_phase(f'structure_{x}.gpw')
        P_c = (phi_c / (2 * np.pi) % 1) 
        
        ## Clean up berryphase files and gs + wf .gpw files.
        remove_files(x)

        P_testa = np.linspace(P_c[0] - n, P_c[0] + n, 2*n + 1)
        P_testb = np.linspace(P_c[1] - n, P_c[1] + n, 2*n + 1)
        P_testc = np.linspace(P_c[2] - n, P_c[2] + n, 2*n + 1) 
       
        if x == path_points_array[0]:
            diffa = np.abs(P_testa - 0)
            diffb = np.abs(P_testb - 0)
            diffc = np.abs(P_testc - 0)
        else:
            diffa = np.abs(P_testa - Pa_path[i-1])
            diffb = np.abs(P_testb - Pb_path[i-1])
            diffc = np.abs(P_testc - Pc_path[i-1])

        ida = np.argmin(diffa)
        idb = np.argmin(diffb)
        idc = np.argmin(diffc)
        
        Pa_path[i] = P_testa[ida] 
        Pb_path[i] = P_testb[idb]  
        Pc_path[i] = P_testc[idc]
        
    ## Change basis from a,b,c, to x,y,z.
    P_v = np.dot(cell_vc, np.array([Pa_path, Pb_path, Pc_path]))
    ## Convert polarizaztion units to nC/m.
    Px_path = P_v[0]*float(1e9)*(_e/V)
    Py_path = P_v[1]*float(1e9)*(_e/V)
    Pz_path = P_v[2]*float(1e9)*(_e/V)
    ## Get the difference in formal polarization between the initial and final structure
    ## for the same branch. This is the spontaneous polarzation. 
    Px = Px_path[-1] - Px_path[0]
    Py = Py_path[-1] - Py_path[0]
    Pz = Pz_path[-1] - Pz_path[0]

    Pa = Pa_path[-1] - Pa_path[0]
    Pb = Pb_path[-1] - Pb_path[0]
    Pc = Pc_path[-1] - Pc_path[0]

    P = np.sqrt(Px**2 + Py**2 + Pz**2)
    E_barrier = E[0] - E[-1] ## Energy barrier between initial and final states. If calculated from -P to P this doesn't make any sense.

    results = {'E_barrier': E_barrier,
               'Pa': Pa,
               'Pb': Pb, 
               'Pc': Pc,
               'Px': Px,
               'Py': Py,
               'Pz': Pz,
               'Ptot': P,
               'E': E,
               'path_points': path_points,
               'Px_path': Px_path,
               'Py_path': Py_path,
               'Pz_path': Pz_path,
               'Pa_path': Pa_path,
               'Pb_path': Pb_path,
               'Pc_path': Pc_path}
    
    return Result(data=results)
 
def get_polarization_direction(data):
    polarpointgroups1 = ['1', '2', 'm', 'mm2']
    polarpointgroups2 = ['3', '3m', '4', '4mm', '6', '6mm']
    rotations = [[[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, -1]],
                 [[-1, 0, 0],
                 [0, 1, 0],
                 [0, 0, -1]],
                 [[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]],
                 [[-1, 1, 0],
                 [0, 1, 0],
                 [0, 0, -1]],
                 [[0, -1, 0],
                 [-1, 0, 0],
                 [0, 0, -1]],
                 [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, -1]]]

    atom = data["structure.json"]
    data = atoms2symmetry(atom, tolerance=0.1, angle_tolerance=1) ## tolerance = 0.1 and angle_tolerance = 0.1 in the old version
    pointgroup = data.dataset['pointgroup']                   
    config = ''
    if pointgroup in polarpointgroups1:
        rot_matrices = data["spglib_dataset"]["rotations"]        
        for x in rot_matrices:
            for y in rotations:
                if np.array_equal(x,y):
                    config = 'in plane'
        if config == '':
            config = '3D'
    if pointgroup in polarpointgroups2:
        config = 'out of plane'

    return config

def get_topological_polarization(rowdata):
    data_pol = rowdata["results-asr.polarization_path.json"]

    fractions = [-5/3, -3/2, -4/3, -1, -2/3, -1/2, -1/3, 0, 1/3, 1/2, 2/3, 1, 4/3, 3/2, 5/3]
    fractions_string = ['-2/3', '-1/2', '-1/3', 0, '-2/3', '-1/2', '-1/3', 0, '1/3', '1/2', '2/3', 0, '1/3', '1/2', '2/3']
    data_pol = read_json(file1)

    Pa = data_pol["Pa_path"][-1]
    distance = abs(fractions - Pa)
    smallest_dist = np.argmin(distance)
    pol_top_a = fractions_string[smallest_dist]

    Pb = data_pol["Pb_path"][-1]
    distance = abs(fractions - Pb)
    smallest_dist = np.argmin(distance)
    pol_top_b = fractions_string[smallest_dist]

    return [pol_top_a, pol_top_b]

def check_polar_symmetry(rowdata):
    polar_pointgroups = [1, 2, 3, 4, 6, "m", "mm2", "3m", "4mm", "6mm"]
    
    data = rowdata["results-asr.structureinfo.json"]
    pointgroup = data["pointgroup"]
    if pointgroup in polar_pointgroups:
        return True
    else:
        return False

def get_parameters(rowdata):
    polarization_data = rowdata[data]
    structure_data = rowdata["results.structureinfo.json"]
    if check_polar_symmetry(structuredata):
        FE = True
        polarization_data = rowdata["results.polarization_path.json"]
        Px = polarization_data["Px"]
        Py = polarization_data["Py"]
        Pz = polarization_data["Pz"]
        Pa = polarization_data["Pa"]
        Pb = polarization_data["Pb"]
        Px = (1e3)*Px ## Convert to pC/m
        Py = (1e3)*Py ## Convert to pC/m
        Pz = (1e3)*Pz ## Convert to pC/m
        if str(get_polarization_direction()[0]) == 'in plane':
            return [FE, Px, Py, Pz, Pa, Pb]

        if str(get_polarization_direction(folder)[0]) == 'out of plane':
            Pz = Pz/2 ## P_{0} (what is calculated is 2P because the path is from -P to P for out of plane materials) 
            pol_top  = get_topological_polarization(polarization_data)
            Pa = pol_top[0]
            Pb = pol_top[1]
        return [FE, Px, Py, Pz, Pa, Pb]

        if str(get_polarization_direction(folder)[0]) == '3D':                
            return [FE, Px, Py, Pz, Pa, Pb]
        else:
            FE = False
            return [FE]
                    
def webpanel(result, row, key_descriptions):
    from asr.database.browser import (table,
                                entry_parameter_description,
                                describe_entry, WebPanel)    
    
    entries = get_parameters(row.data)
    if entries[0]:
        parameter_description = entry_parameter_description(
            row.data,
            'asr.polarization_path')
        explanation_FE = ('The material is ferroelectric\n\n' + parameter_description)
        explanation_Px = ('The spontaneous polarization along the x-direction\n\n' + parameter_description)
        explanation_Py = ('The spontaneous polarization along the x-direction\n\n' + parameter_description)
        explanation_Pz = ('The spontaneous polarization along the x-direction\n\n' + parameter_description)
        explanation_Pa = ('The spontaneous polarization along the x-direction\n\n' + parameter_description)
        explanation_Pb = ('The spontaneous polarization along the x-direction\n\n' + parameter_description)

        FE = describe_entry('Ferroelectric', description=explanation_FE)
        Px = describe_entry('Px', description=explanation_Px)
        Py = describe_entry('Py', description=explanation_Py)
        Pz = describe_entry('Pz', description=explanation_Pz)
        Pa = describe_entry('Pa', description=explanation_Pa)
        Pb = describe_entry('Pb', description=explanation_Pb)

        polarization_table = table(result, 'Property',
                                   [FE, Px, Py, Pz, Pa, Pb],
                                   kd=key_descriptions)

        from asr.utils.hacks import gs_xcname_from_row
        panel = WebPanel(title=f'Basic electronic properties ({xcname})',
                         columns=[[polarization_table], []],
                         sort=11)
        return [panel]

    if not entries[0]:
        parameter_description = entry_parameter_description(
            row.data,
            'asr.polarization_path')
        explanation_FE = ('The material is pyroelectric\n\n' + parameter_description)
        
        FE = describe_entry('Ferroelectric', description=explanation_FE)
        
        polarization_table = table(result, 'Property',
                                   [FE],
                                   kd=key_descriptions)

        from asr.utils.hacks import gs_xcname_from_row
        panel = WebPanel(title=f'Basic electronic properties ({xcname})',
                         columns=[[polarization_table], []],
                         sort=11)
        return [panel]
    
@prepare_result
class Result(ASRResult):
    """Container for formal polarization results."""

    E_barrier: float
    Pa: float
    Pb: float
    Pc: float
    Px: float
    Py: float
    Pz: float
    Ptot: float
    E: np.ndarray
    path_points: np.ndarray
    Pa_path: np.ndarray
    Pb_path: np.ndarray
    Pc_path: np.ndarray
    Px_path: np.ndarray
    Py_path: np.ndarray
    Pz_path: np.ndarray
    
    key_descriptions = dict(
        E_barrier='Energy barrier between non polar and polar structure [eV].',
        Pa="Dimensionless spontaneous polarization along lattice vector a",
        Pb='Dimensionless spontaneous polarization along lattice vector b',
        Pc='Dimensionless spontaneous polarization along lattice vector c',
        Px='Spontaneous polarization along the x-direction [nC/m]',
        Py='Spontaneous polarization along the y-direction [nC/m]',
        Pz='Spontaneous polarization along the z-direction [nC/m]',
        Ptot='Total spontaneous polarization [nC/m]',
        E='Ground state energy for each structure along the adiabatic path [eV]',
        path_points='Reaction coordinates used for the adiabatic path',
        Pa_path='Branch fixed value of the dimensionless formal polarization along lattice vector a',
        Pb_path='Branch fixed value of the dimensionless formal polarization along lattice vector a',
        Pc_path='Branch fixed value of the dimensionless formal polarization along lattice vector a',
        Px_path='Branch fixed value of the formal polarization along the x-direction [nC/m]',
        Py_path='Branch fixed value of the formal polarization along the y-direction [nC/m]',
        Pz_path='Branch fixed value of the formal polarization along the z-direction [nC/m]',
    )

    formats = {'ase_webpanel': webpanel}

if __name__ == '__main__':
    main.cli()
