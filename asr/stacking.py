"""Template recipe."""
import json
from pathlib import Path
from ase.io import read
from asr.core import command, option
from math import isclose
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

@command('asr.setup.stackings',
         known_exceptions=known_exceptions,
         tests=tests,
         resources='24:10h')
@option('--distance', type=float,
        help='Initial distance between the monolayers')
def main(distance=12.):
    """
    Creates bilayer structures.

    By default, this recipe takes the atomic structure in 'structure.json'

    Saves it in a subfolder that it creates under the material folder.

    TODO: - remove loop of going through all materials folders and just write
            recipe for one material only. The looping should be a task of the
            workflow and myqueue
          - put get_magstate function outsinde of main
          - get rid of hardcoded parameters in create_stackings
    """
    atom = read('structure.json')

    try:
        magstate = get_magstate(atom)
    except RuntimeError:
        magstate = nm 

    create_stackings(magstate, distance, atom)


def get_magstate(atom):
    """
    Obtains the magnetic state of a given atomic structure
    """
    magmom = atom.get_magnetic_moment()
    magmoms = atom.get_magnetic_moments()

    if abs(magmom) > 0.02:
        state = 'fm'
    elif abs(magmom) < 0.02 and abs(magmoms).max() > 0.1:
        state = 'afm'
    else:
        state = 'nm'

    return state


def cell_check(stack, tol=1e-5):
    """
    Finds the material's unit cell type and returns it

    TODO: - think of suitable value for the tolerances
    """
    unit_cell = stack.get_cell_lengths_and_angles()
    if isclose(unit_cell[0],unit_cell[1], abs_tol=tol) and abs(unit_cell[5] - 90) < tol:
    	unit_cell_type = 'square'
    elif round(unit_cell[3]) == 90 and abs(unit_cell[5] - 90) < tol:
    	unit_cell_type = 'rectangular'
    elif isclose(unit_cell[0], unit_cell[1], abs_tol=tol) and ((120 - unit_cell[5]) < tol or abs(60 - unit_cell[5]) < tol):
    	unit_cell_type = 'hexagonal'
    elif not isclose(unit_cell[0], unit_cell[1], abs_tol=tol) and abs(90 - unit_cell[5]) > tol and unit_cell[0]/2 - abs(unit_cell[1]*math.cos(unit_cell[5]*math.pi/180)) < tol:
    	unit_cell_type = 'centered'
    else:
    	unit_cell_type = 'oblique'
    
    return unit_cell_type


def rot_label(final_rotations):
    """
    A naming convention we made to keep track of what rotations had been performed. 
    To be frank I would probably remove this and find a better system for it
    """
    labels = []
    for i in range(len(final_rotations)):
    	rot_lab = '0-NNNN'
    	w, v = np.linalg.eig(final_rotations[i])
    	det = np.linalg.det(final_rotations[i])
    	eig_vec1 = v[0]
    	eig_vec2 = v[1]
    	theta = np.arccos(np.dot(eig_vec1,eig_vec2)/(np.linalg.norm(eig_vec1)*np.linalg.norm(eig_vec2)))
    	rot_mat = final_rotations[i]
            
    	# The code is basically a bunch of if statements as you could find the rotation performed from the rotation matrix
    	# and if it had a certain element it meant a certain rotation had been performed 
    	# eg. last element is -1 means there was a reflection in the Z-axis
    	if rot_mat[2][2] == -1:
    		rot_lab = rot_lab[:4] + 'Z' + rot_lab[5:]
    	if round(v[0][0],3) == 0.707 and np.iscomplex(w).any()==False or round(v[0][1],3) == 0.894 or round(v[0][1],3) == 0.447:
    		rot_lab = rot_lab[:5] + 'D' + rot_lab[6:]
    	if round(v[0][1],3) == 0.894:
    		rot_lab =  '120' + rot_lab[1:]
    	elif round(v[0][1],3) == 0.447:
    		rot_lab = '60' + rot_lab[1:]
    	elif np.iscomplex(w).any()==True:
    		if rot_mat[0][0] != 0 and rot_mat[1][0] != 0:
    			theta_trace = 60
    		elif rot_mat[1][0] != 0 and rot_mat[1][1] != 0:
    			theta_trace = 120
    		else:
    			argument = (rot_mat[0][0]+rot_mat[1][1]+rot_mat[2][2]-1)/2
    			if argument == -1.5:
    				theta_trace = 60 
    			else:
    				theta_trace = np.arccos(argument)
    				theta_trace = int(round(theta_trace*180/pi))
    			if theta_trace == 180:
    				theta_trace = 90
    		rot_lab = str(theta_trace) + rot_lab[1:]
    	if rot_lab[0] == '0' and np.sum(rot_mat[:,1]) < 0:
    		index = rot_lab.index('N')
    		rot_lab = rot_lab[:index] + 'X' + rot_lab[index+1:]
    	elif '90' in rot_lab and rot_mat[0][1] == 1 and rot_mat[1][0] == -1:
    		index = rot_lab.index('N')
    		rot_lab = rot_lab[:index] + 'X' + rot_lab[index+1:]
    	elif '120' in rot_lab and rot_mat[0][1] == 1 and rot_mat[1][0] == -1 and 'D' not in rot_lab:
    		index = rot_lab.index('N')
    		rot_lab = rot_lab[:index] + 'X' + rot_lab[index+1:]
    	elif '120' in rot_lab and rot_mat[1][0] == 1 and 'D' in rot_lab:
    		index = rot_lab.index('N')
    		rot_lab = rot_lab[:index] + 'X' + rot_lab[index+1:]
    	elif '60' in rot_lab and rot_mat[0][1] == 1 and rot_mat[1][0] == -1 and 'D' not in rot_lab:
    		index = rot_lab.index('N')
    		rot_lab = rot_lab[:index] + 'X' + rot_lab[index+1:]
    	elif '60' in rot_lab and rot_mat[0][1] == -1 and 'D' in rot_lab:
    		index = rot_lab.index('N')
    		rot_lab = rot_lab[:index] + 'X' + rot_lab[index+1:]
    	
    	# y-reflection check
    	if rot_lab[0] == '0' and np.sum(rot_mat[:,0]) < 0:
    		if 'X' in rot_lab:
    			index = rot_lab.index('N')
    		else:
    			index_list = rot_lab.split('N', 1)
    			index = len(index_list[0])+1
    		rot_lab = rot_lab[:index] + 'Y' + rot_lab[index+1:]
    	elif '90' in rot_lab and rot_mat[0][1] == 1 and rot_mat[1][0] == -1:
    		if 'X' in rot_lab:
    			index = rot_lab.index('N')
    		else:
    			index_list = rot_lab.split('N', 1)
    			index = len(index_list[0])+1
    		rot_lab = rot_lab[:index] + 'Y' + rot_lab[index+1:]
    	elif '120' in rot_lab and rot_mat[0][1] == 1 and rot_mat[1][0] == -1 and 'D' not in rot_lab:
    		if 'X' in rot_lab:
    			index = rot_lab.index('N')
    		else:
    			index_list = rot_lab.split('N', 1)
    			index = len(index_list[0])+1
    		rot_lab = rot_lab[:index] + 'Y' + rot_lab[index+1:]
    	elif '120' in rot_lab and rot_mat[1][0] == 1 and 'D' in rot_lab:
    		if 'X' in rot_lab:
    			index = rot_lab.index('N')
    		else:
    			index_list = rot_lab.split('N', 1)
    			index = len(index_list[0])+1
    		rot_lab = rot_lab[:index] + 'Y' + rot_lab[index+1:]
    	elif '60' in rot_lab and rot_mat[0][1] == 1 and rot_mat[1][0] == -1 and 'D' not in rot_lab:
    		if 'X' in rot_lab:
    			index = rot_lab.index('N')
    		else:
    			index_list = rot_lab.split('N', 1)
    			index = len(index_list[0])+1
    		rot_lab = rot_lab[:index] + 'Y' + rot_lab[index+1:]
    	elif '60' in rot_lab and rot_mat[0][1] == -1 and 'D' in rot_lab:
    		if 'X' in rot_lab:
    			index = rot_lab.index('N')
    		else:
    			index_list = rot_lab.split('N', 1)
    			index = len(index_list[0])+1
    		rot_lab = rot_lab[:index] + 'Y' + rot_lab[index+1:]
    	labels.append(rot_lab)
    return labels	

def translation(x_trans, y_trans, d, new_mat, layer_mat):
	"""
	Performs the translation. Input is the translations desired
	new_mat is the rotated material, and layer_mat the original material
	"""
	structure = layer_mat.copy()
	copy_mat = new_mat.copy()
	# Translates both in x and y, but also in z with the distance d
	copy_mat.translate([x_trans,y_trans,d])
	structure += copy_mat
	structure.wrap()
	return structure

def rot_trans_append(x,y,rotation_types,rotation_translation_types):
	"""
	Adds the translations performed to the naming convention
	"""
	rotation_types_copy = rotation_types + '-' + str(x) + str(y)
	rotation_translation_types.append(rotation_types_copy)
	return rotation_translation_types

def rotations(layer_mat):
	"""
	Finds all the possible types of rotations possible for the material
	Returns a list with the rotated material, and a corresponding list with
	a naming convention for the performed rotations
	"""
	mat_new = layer_mat.copy()
	# The possible rotations are found by finding all symmetry operations for a single atom
	atoms = Atoms('C', positions=[[0,0,0]])
	atoms.set_cell(layer_mat.get_cell())
	# Uses SPGlib to find the symmetries
	symmetryC = spglib.get_symmetry(atoms)

	final_rotations = symmetryC['rotations']
	final_translations = symmetryC['translations']
	
	# Performs the rotations on the material and saves each one in a list
	liste = []
	for an in range(len(final_rotations)):
		U_cc = final_rotations[an]
		t_c = final_translations[an]
		new_layer = layer_mat.copy()
		spos_ac = new_layer.get_scaled_positions()
		spos_ac = np.dot(spos_ac, U_cc.T) + t_c
		new_layer.set_scaled_positions(spos_ac)
		new_layer.wrap(pbc=[1,1,1])
		liste.append(new_layer)

	# Finds the label for the performed rotation - see rot_label function
	rotation_types = rot_label(final_rotations)

	# Removes all identical performed rotations
	for i in range(len(liste)):
		# Counter is so that when j loops through the second list, if an element was removed
		# it correspondingly selects one less in the list
		counter = 0
		for j in range(len(liste)):
			if j > i:
				j -= counter
				# If all the positions are the same the rotation is removed
				dist = ((liste[i].positions - liste[j].positions)**2).sum()
				if dist < 1e-5:
					liste.pop(j)
					rotation_types.pop(j)
					counter += 1
	return liste, rotation_types

def build_layers(d, layer_mat, unit_cell_type, rotation_transform, rotation_types, unit_cell):
	"""
	Builds the bilayer. Input is the distance, which is currently hardcoded to 12, the material, the unit cell type (the string from the function)
	the performed rotations, the naming convention of the rotations, and finally unit cell variable is a list of unit cell lengths and angles
	"""
	structure_list = []
	rotation_translation_types = []
	positions = layer_mat.get_positions()
	
	a = unit_cell[0]
	b = unit_cell[1]
	c = unit_cell[2]
	
	# Different translations are performed depending on the unit cell type
	# For the hexagonal, the hexagonal grid is slided two times - see our paper for more information (page 15)
	if unit_cell_type == 'hexagonal': 
		# Performs the translation for every rotation found earlier
		for q in range(len(rotation_transform)):
			# Performs the translation
			structure1 = translation(0,0,d,rotation_transform[q],layer_mat)
			# Adds translation performed to naming convention
			rotation_translation_types = rot_trans_append(0,0,rotation_types[q],rotation_translation_types)

			structure2 = translation(positions[1][0],positions[1][1],d,rotation_transform[q],layer_mat)
			rotation_translation_types = rot_trans_append(1,1,rotation_types[q],rotation_translation_types)
			structure3 = translation(2*positions[1][0],2*positions[1][1],d,rotation_transform[q],layer_mat)
			rotation_translation_types = rot_trans_append(2,2,rotation_types[q],rotation_translation_types)
			structure_list.append(structure1)
			structure_list.append(structure2)
			structure_list.append(structure3)

	# For all the other unit cells a different type of translation is performed
	if unit_cell_type == 'oblique' or unit_cell_type == 'rectangular' or  unit_cell_type == 'square' or unit_cell_type == 'centered': 
		for q in range(len(rotation_transform)):
			structure1 = translation(0,0,d,rotation_transform[q],layer_mat)
			rotation_translation_types = rot_trans_append(0,0,rotation_types[q],rotation_translation_types)
			structure2 = translation(1/2*a,0,d,rotation_transform[q],layer_mat)
			rotation_translation_types = rot_trans_append(1,0,rotation_types[q],rotation_translation_types)
			structure3 = translation(0,1/2*b,d,rotation_transform[q],layer_mat)
			rotation_translation_types = rot_trans_append(0,1,rotation_types[q],rotation_translation_types)
			structure4 = translation(1/2*a,1/2*b,d,rotation_transform[q],layer_mat)
			rotation_translation_types = rot_trans_append(1,1,rotation_types[q],rotation_translation_types)
			structure_list.append(structure1)
			structure_list.append(structure2)
			structure_list.append(structure3)
			structure_list.append(structure4)

	# Removes identical structures
	for i in range(len(structure_list)):
		counter = 0
		for j in range(len(structure_list)):
			check = 0
			if j > i:
				j -= counter
				atoms1 = structure_list[i]
				atoms1_positions = atoms1.get_positions()
				atom_type1 = atoms1.get_atomic_numbers()
				atoms2 = structure_list[j]
				atoms2_positions = atoms2.get_positions()
				atom_type2 = atoms2.get_atomic_numbers()
				# Removes them by checking every position
				for k in range(len(atoms1)):
					for q in range(len(atoms2)):
						if (np.sqrt((atoms1_positions[k] - atoms2_positions[q])**2)).sum() < 10e-5:
							if atom_type1[k] == atom_type2[q]:
								check += 1
				# If all positions are identical, remove it
				if check == len(atoms1):
					structure_list.pop(j)
					rotation_translation_types.pop(j)
					counter += 1
	return structure_list, rotation_translation_types

def create_stackings(magnetism, saves, d, atom):	
	"""
	This function starts the other ones that create the bilayer stackings and saves the results
	The parameters are the magnetism that it found earlier, saves is an outdated function that was used for testing (can be removed),
	d is the initial distance between the two layers, and atom is the the read material
	"""
	magnetism = magnetism.upper()
	cwd = os.getcwd()
	folder = cwd
	#original_npy_folder = 'bilayer/{}-{}/'.format(prototype, formula)
	#layer_mat = read(original_npy_folder + '{}-{}-{}.xyz'.format(prototype, formula, magnetism))

	# Just renaming a variable here as it had difference name in some parts of the code	
	layer_mat = atom
	# Finds the unit cell of the material - see cell_check function
	unit_cell = layer_mat.get_cell_lengths_and_angles()
	unit_cell_type = cell_check(layer_mat, tol=1e-5)
	# Finds the rotations that are possible - see rotations function
	mat_rot, rotation_types = rotations(layer_mat)
	# Builds the bilayer - see build_layers function
	structure_list, rotation_translation_types = build_layers(d, layer_mat, unit_cell_type, mat_rot, rotation_types,unit_cell)
	
	if saves == True:	
		for rotation_type in rotation_translation_types:
			try:
				if not os.path.exists(folder + '/{}-{}'.format(magnetism,rotation_type)):
					os.mkdir(folder + '/{}-{}'.format(magnetism,rotation_type))
			except:
				pass
	else:
		rotation_type = saves
		try:
			if not os.path.exists(folder + '/{}-{}'.format(magnetism,rotation_type)):
				os.mkdir(folder + '/{}-{}'.format(magnetism,rotation_type))
		except:
			pass


	for i in range(len(structure_list)):
		if saves == True:
			#write(folder + '/{}-{}'.format(magnetism,rotation_translation_types[i])+'/{}-{}.json'.format(magnetism,rotation_translation_types[i]), structure_list[i],format='json')
			write(folder + '/{}-{}'.format(magnetism,rotation_translation_types[i])+'/unrelaxed_structure.json', structure_list[i],format='json')
		elif rotation_translation_types[i] == saves:
			#write(folder + '/{}-{}'.format(magnetism,rotation_translation_types[i])+'/{}-{}.json'.format(prototype,formula,magnetism,rotation_translation_types[i]), structure_list[i],format='json')
			write(folder + '/{}-{}'.format(magnetism,rotation_translation_types[i])+'/unrelaxed_structure.json', structure_list[i],format='json')


if __name__ == '__main__':
    main()
