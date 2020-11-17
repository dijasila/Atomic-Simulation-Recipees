import argparse
import numpy as np
from ase.db import connect
from tqdm import tqdm
from pathlib import Path
from ase.io.jsonio import write_json
from asr.core import command, option




pi = np.pi
rad2deg = 180 / pi



# Angle between v1 and v2, measured counter-clockwise.
# If angle is > 0, you have to rotate v1 counter-clockwise to match v2, and vv.sa
def angle_between(v1, v2):
    v1u = v1 / np.linalg.norm(v1)
    v2u = v2 / np.linalg.norm(v2)
    if v2u[1] >= 0 and v1u[1] >= 0:
        return np.arccos(v2u[0]) - np.arccos(v1u[0])
    if v2u[1] <= 0 and v1u[1] <= 0:
        return np.arccos(v1u[0]) - np.arccos(v2u[0])
    if v2u[1] >= 0 and v1u[1] <= 0:
        return np.arccos(v1u[0]) + np.arccos(v2u[0])
    if v2u[1] <= 0 and v1u[1] >= 0:
        return - np.arccos(v1u[0]) - np.arccos(v2u[0])




# Rotate vector counter-clockwise for positive angles, and vv.sa
def rot_vec(v, th):
    rot_matrix = [ [np.cos(th), -np.sin(th)],    
                   [np.sin(th), np.cos(th)] ]
    return np.dot(rot_matrix, v)




# Obtain new vector as linear combination of two basis vectors
def lc_vec(m1, m2, v1, v2):
    return [ m1*v1[0] + m2*v2[0], m1*v1[1] + m2*v2[1] ]




# Obtain vector modulus
def norm(v):
    return np.linalg.norm(v)




# Obtain area of the parallelogram defined by two vectors
def area(v1, v2):
    return abs(v1[0]*v2[1] - v1[1]*v2[0])




# Return absolute value of the angle between two vectors, between 0° and 180°
def AngIntern(v1, v2):
    val_angle = angle_between(v1, v2)
    if abs(val_angle) > np.pi:
        return abs(2*pi - val_angle)
    else: 
        return abs(val_angle)




# Check for duplicate cells and select the best one
def best_duplicate(val_twist, vec_twist, val_strain, vec_strain, val_area, vec_area, ang1, vec_angle):

    # Add the first area to the comparison vector. [1, -1] means "Add"
    if len(vec_twist) == 0:
        return [1, -1]

    i = 0
    while i <= len(vec_twist) - 1:

        # If the two supercells have identical twist angle, start deciding which one is the best
        if abs(val_twist - vec_twist[i]) < 1.0e-6:

            # keep the one with lower strain. [-1, i] means "subsitute supercell i with the current one"
            if vec_strain[i] - val_strain > 0.1:
                    return [-1, i]
            else: 
                # If strains are identical, keep the one with lower number of atoms
                if val_area < vec_area[i]:
                    return [-1, i]
                else:
                    ang2 = vec_angle[i]
                    # If they have also the same number of atoms, keep the one with higher internal angle
                    if ang2 < pi/2 and ang1 < pi-ang2 and ang1 > ang2:
                        return [-1, i]
                    elif ang2 > pi/2 and ang1 < ang2 and ang1 > pi - ang2:
                        return [-1, i]
                    else:
                        # If none of the above conditions is satisfied, discard the current supercell
                        return [0, 0]

        # If the twist angles are different, go to the next supercell and restart the tests
        else: 
            i += 1
            
    # No match after scanning all comparison vector: add as new result
    return [1, -1]




# Sort "vec" based on the values contained in "sort_vec"
def CustomSort(vec, sort_vec):
    return [ vec[i] for i in np.argsort(sort_vec) ]




def MatchCells(lyr_a, lyr_b, workdir, max_coef, tol_theta, store_all, scan_all, sort, max_strain, max_number_of_atoms, min_internal_angle, max_internal_angle):

    # ----------------- DEFINING LATTICES  ------------------

    # Pull down cell info for the two layers from C2DB
    db = connect('/home/niflheim/steame/hetero-bilayer-project/databases/gw-bulk.db')

    layer_a = db.get(uid=lyr_a)
    cell_a = layer_a.cell
    name_a = layer_a.formula
    a1 = cell_a[0, 0:2]
    a2 = cell_a[1, 0:2]
    starting_area_a = area(a1, a2)
    
    layer_b = db.get(uid=lyr_b)
    cell_b = layer_b.cell
    name_b = layer_b.formula
    b1 = cell_b[0, 0:2]
    b2 = cell_b[1, 0:2]
    starting_area_b = area(b1, b2)
    
    
    
    # ----------------- DEFINING PARAMETERS  ------------------
    
    # Maximum coefficient value for the LCs
    max_coeff = max_coef

    # Tolerance over relative rotation angle in radians
    pi = np.pi
    rad2deg = 180 / pi

    tol_theta = tol_theta / rad2deg

    # For being sure that linear independence is verified for finite precision
    tol_area = 1

    # Initializing stuff
    vectors_a, moduli_a, indexes_a = [], [], []
    vectors_b, moduli_b, indexes_b = [], [], []
    max_a = 0
    max_b = 0 
    n_max = 1       
    increase = 1
    matches = []
    
    
    #------------- GENERATING LINEAR COMBINATIONS -------------
    
    # Scan either the upper half of the XY plane or all of it.
    if bool(scan_all) == True:
        m_lower = - max_coeff
        n_lower = - n_max
    else:
        m_lower = 0
        n_lower = 0
    
    
    # Generate all relevant LCs for lattice A with m1, m2 <= max_coeff
    for m1 in range( max_coeff, - max_coeff - 1, -1 ):
        for m2 in range( m_lower, max_coeff + 1):
            # Exclude the origin
            if [m1, m2] != [0, 0]:
                vec = lc_vec(m1, m2, a1, a2)
                vectors_a.append(vec)
                moduli_a.append(norm(vec))
                indexes_a.append([m1, m2])
                # Store the maximum modulus values in order to set the number of LCs for lattice B
                if abs(m1) == max_coeff and m2 == max_coeff and norm(vec) > max_a:
                    max_a = norm(vec)
    
    # Generate LCs for lattice B inside shells with increasing n1,n2 values
    while increase == 1:
        for n1 in range( n_max, - n_max - 1, -1 ):
            for n2 in range( n_lower, n_max + 1 ):
                if [n1, n2] != [0, 0]:
                    vec = lc_vec(n1, n2, b1, b2)
                    vectors_b.append(vec)
                    moduli_b.append(norm(vec))
                    indexes_b.append([n1, n2])
                    if abs(n1) == n_max and n2 == n_max and norm(vec) > max_b:
                        max_b = norm(vec)
        # Once a shell is completed, compare the max moduli for lattices A and B.
        # Stop when the ones for lattice B are larger.
        if max_b > max_a: 
            increase = 0
        else:
            n_max += 1
            vectors_b, moduli_b, indexes_b = [], [], []
    
    
    #------------------ MATCHING OVER MODULI ------------------
    
    match_a, match_index_a, match_b, match_index_b = [], [], [], []
    angles = []
    strains = []
    
    # Store the LCs in lattice A and B with matching moduli
    for i in range(len(moduli_a)):
        for j in range(len(moduli_b)):
            strain_ij = max([abs(moduli_a[i] - moduli_b[j]) / moduli_a[i] * 100,
                             abs(moduli_a[i] - moduli_b[j]) / moduli_b[j] * 100])
            if strain_ij < max_strain:
                angle_ij = angle_between(vectors_a[i], vectors_b[j])
                match_a.append(vectors_a[i])
                match_b.append(vectors_b[j])
                match_index_a.append(indexes_a[i])
                match_index_b.append(indexes_b[j])
                angles.append(angle_ij)
                strains.append(strain_ij)
    
    
    #------------------ FINDING SUPERCELLS --------------------
    
    match_num = len(match_a)
    superc_a1, superc_a2, superc_b1, superc_b2 = [], [], [], []
    a1_coeffs, b1_coeffs, a2_coeffs, b2_coeffs, = [], [], [], []
    angles_a1b1, angles_a2b2 = [], []
    angles_intern = []
    areas_a, areas_b = [], []
    ratios_a, ratios_b = [], []
    natoms = []
    strain_ab = []
    
    # Generate supercells by finding all the linearly independent LCs (area = 0)
    # which share the same rotation angle. 
    if match_num > 2500:
        print("\nI found a lot of matching vectors! This will take some time, go have a coffee...\n")
    else:
        print("\nGenerating supercells...\n")
    
    for i in tqdm(range(match_num)):
    
        for j in range(i + 1, match_num):

            # For commensurate lattices, we will always obtain all the multiples of the  
            # superposition of the unit cells. Let's discard them, and keep only the first superposition.
            if  [match_index_a[i], match_index_b[i]] == [[1, 0], [1, 0]] or match_index_a[i] != match_index_b[i]:
                ratio_a = area(match_a[i], match_a[j]) / starting_area_a
                ratio_b = area(match_b[i], match_b[j]) / starting_area_b
                natom = ratio_a * layer_a.natoms + ratio_b * layer_b.natoms
                test_intern = AngIntern(match_b[i], match_b[j])
                test_area = area(match_b[i], match_b[j])
                test_twist = angles[i]

                if abs(angles[i] - angles[j]) <= tol_theta and test_intern >= min_internal_angle and test_intern <= max_internal_angle and natom <= max_number_of_atoms:

                    max_strain_ij = max(strains[i], strains[j])
                    check = best_duplicate(test_twist, angles_a1b1, max_strain_ij, strain_ab, test_area, areas_b, test_intern, angles_intern)
    
                    # Save new supercells
                    if check == [1, -1] or store_all == True:
                        superc_a1.append(match_a[i])
                        superc_a2.append(match_a[j])
                        superc_b1.append(match_b[i])
                        superc_b2.append(match_b[j])
                        a1_coeffs.append(match_index_a[i]) 
                        a2_coeffs.append(match_index_a[j])
                        b1_coeffs.append(match_index_b[i])
                        b2_coeffs.append(match_index_b[j])
                        angles_a1b1.append(angles[i])
                        angles_a2b2.append(angles[j])
                        angles_intern.append(test_intern)
                        areas_a.append(area(match_a[i], match_a[j]))
                        areas_b.append(area(match_b[i], match_b[j]))
                        ratios_a.append(ratio_a)
                        ratios_b.append(ratio_b)
                        natoms.append(natom)
                        strain_ab.append(max_strain_ij)
                       
    
                    # Replace equivalent supercells if one with a smaller rotation angle is found.
                    if check[0] == -1 and store_all == False:
                        superc_a1[check[1]] = match_a[i]
                        superc_a2[check[1]] = match_a[j]
                        superc_b1[check[1]] = match_b[i]
                        superc_b2[check[1]] = match_b[j]
                        a1_coeffs[check[1]] = match_index_a[i]
                        a2_coeffs[check[1]] = match_index_a[j]
                        b1_coeffs[check[1]] = match_index_b[i]
                        b2_coeffs[check[1]] = match_index_b[j]
                        angles_a1b1[check[1]] = angles[i]
                        angles_a2b2[check[1]] = angles[j]
                        angles_intern[check[1]] = test_intern
                        areas_a[check[1]] = area(match_a[i], match_a[j])
                        areas_b[check[1]] = area(match_b[i], match_b[j])
                        ratios_a[check[1]] = ratio_a
                        ratios_b[check[1]] = ratio_b
                        natoms[check[1]] = natom
                        strain_ab[check[1]] = max_strain_ij
    
    
    # Sort results according to strain / number of atoms
    if sort == "strain":
        natoms = CustomSort(natoms, strain_ab)
        ratios_a = CustomSort(ratios_a, strain_ab)
        ratios_b = CustomSort(ratios_b, strain_ab)
        a1_coeffs = CustomSort(a1_coeffs, strain_ab)
        a2_coeffs = CustomSort(a2_coeffs, strain_ab)
        b1_coeffs = CustomSort(b1_coeffs, strain_ab)
        b2_coeffs = CustomSort(b2_coeffs, strain_ab)
        angles_a1b1 = CustomSort(angles_a1b1, strain_ab)
        angles_intern = CustomSort(angles_intern, strain_ab)
        strain_ab = CustomSort(strain_ab, strain_ab)
    
    else:
        ratios_a = CustomSort(ratios_a, natoms)
        ratios_b = CustomSort(ratios_b, natoms)
        a1_coeffs = CustomSort(a1_coeffs, natoms)
        a2_coeffs = CustomSort(a2_coeffs, natoms)
        b1_coeffs = CustomSort(b1_coeffs, natoms)
        b2_coeffs = CustomSort(b2_coeffs, natoms)
        angles_a1b1 = CustomSort(angles_a1b1, natoms)
        angles_intern = CustomSort(angles_intern, natoms)
        strain_ab = CustomSort(strain_ab, natoms)
        natoms = CustomSort(natoms, natoms)
    

    nsol = len(areas_a)
    print("\n{0} supercells found! Bye Bye...\n".format(nsol))


    
    # ------------------- SAVE TO READABLE FORMAT ---------------------
    

    file_cells = f"{workdir}/moirecells.cells"
    with open(file_cells, "w") as log:
    
        print("Layer A unique identifier:", file=log)
        print(layer_a.uid, '\n', file=log)
        print("Layer B unique identifier:", file=log)
        print(layer_b.uid, '\n', file=log)
        print("Number of solutions:", file=log)
        print(len(areas_a), '\n', file=log)
        
        print("--------------------------------------------------- SUPERCELLS --------------------------------------------------------\n", file=log)
        print("{:>3}{:>9}{:>16}{:>8}{:>5}{:>6}{:>5}{:>8}{:>5}{:>6}{:>5}{:>17}{:>15}{:>11}\n".format("#", "Atoms", "Surf. Ratios", "m1", "m2", "m1'", "m2'", "n1", "n2", "n1'", "n2'", "Angle(intern)", "Angle(twist)", "Strain(%)"), file=log)
        
        for i in range(nsol):
                print("{:3}".format(i), end='', file=log)
                print("{:8.0f}".format(natoms[i]), end='', file=log)
                print("{:>10.0f}".format(ratios_a[i]), end='', file=log)
                print("{:>5.0f}".format(ratios_b[i]), end='', file=log)
                print("{:>10}".format(a1_coeffs[i][0]), end='', file=log)
                print("{:>5}".format(a1_coeffs[i][1]), end='', file=log)
                print("{:>5}".format(a2_coeffs[i][0]), end='', file=log)
                print("{:>5}".format(a2_coeffs[i][1]), end='', file=log)
                print("{:>9}".format(b1_coeffs[i][0]), end='', file=log)
                print("{:>5}".format(b1_coeffs[i][1]), end='', file=log)
                print("{:>5}".format(b2_coeffs[i][0]), end='', file=log)
                print("{:>5}".format(b2_coeffs[i][1]), end='', file=log)
                print("{:>16.6f}".format(angles_intern[i] * rad2deg), end='', file=log)
                print("{:>16.6f}".format(angles_a1b1[i] * rad2deg), end='', file=log)
                print("{:>11.4f}\n".format(strain_ab[i]), end='', file=log)


    # ------------------- SAVE TO JSON ---------------------
     
    results = {} 
    results["uid_a"] = layer_a.uid
    results["uid_b"] = layer_b.uid
    results["number_of_solutions"] = nsol
    results["solutions"] = {} 

    for i in range(nsol):
        results["solutions"][f"{i}"] = {}
        results["solutions"][f"{i}"]["natoms"] = natoms[i]
        results["solutions"][f"{i}"]["indexes_a"] = [a1_coeffs[i][0], a1_coeffs[i][1], 
                                                     a2_coeffs[i][0], a2_coeffs[i][1]]
        results["solutions"][f"{i}"]["indexes_b"] = [b1_coeffs[i][0], b1_coeffs[i][1], 
                                                     b2_coeffs[i][0], b2_coeffs[i][1]]
        results["solutions"][f"{i}"]["internal_angle"] = angles_intern[i] * rad2deg
        results["solutions"][f"{i}"]["twist_angle"] = angles_a1b1[i] * rad2deg
        results["solutions"][f"{i}"]["max_strain"] = strain_ab[i]
            
    file_json = f"{workdir}/moirecells.json"
    write_json(file_json, results)



@command('asr.findmoire')
@option('--max-coef', type=int, help='Max coefficient for linear combinations of the starting vectors')
@option('--tol-theta', type=float, help='Tolerance over rotation angle difference between matching vector pairs')
@option('--store-all', type=bool, help='True: store all the possible matches. False: store only unique supercell')
@option('--scan-all', type=bool, help='True: scan linear combinations in all the XY plane. False: scan only the upper half')
@option('--sort', type=str, help='Sort results by number of atoms or max strain')
@option('--max-strain', type=float, help='Store only supercells with max (percent) strain lower than the specified one')
@option('--max-number-of-atoms', type=float, help='Store only supercells with lower number f atoms than the specified one')
@option('--min-internal-angle', type=float, help='Lower limit for the supercell internal angle (in degrees)')
@option('--max-internal-angle', type=float, help='Upper limit for the supercell internal angle (in degrees)')
@option('--overwrite', type=bool, help='True: Regenerate directory structure overwriting old files; False: generate results only for new entries')
@option('--database', type=str, help='Path of the .db database file for retrieving structural information')
@option('--uids', type=str, help='Path of the file containing the unique ID list of the materials to combine')
def main(max_coef: int = 10, tol_theta: float = 0.05, store_all: bool = False, scan_all: bool = False, sort: str = "natoms", max_strain: float = 1.0, max_number_of_atoms: int = 300, min_internal_angle: float = 30.0, max_internal_angle: float = 150.0, overwrite: str = False, database: str = "/home/niflheim/steame/hetero-bilayer-project/databases/gw-bulk.db", uids: str = "/home/niflheim/steame/venvs/hetero-bilayer-new/venv/asr/asr/test/moire/tree/uids"):

    db = connect(database)

    with open("/home/niflheim/steame/venvs/hetero-bilayer-new/venv/asr/asr/test/moire/tree/uids", "r") as f:
        monos = [ i.split()[0] for i in f.readlines() ]

    for i in range(len(monos)):
        for j in range(i, len(monos)):
            name_a = db.get(uid=monos[i]).formula
            name_b = db.get(uid=monos[j]).formula
            workdir = f"{name_a}-{name_b}"
            dirwork = f"{name_b}-{name_a}"

            # Generate directory and results for the current bilayer if they don't exist
            # or "overwrite" option is passed"
            if Path(workdir).exists() == False and Path(dirwork).exists() == False:
                Path(workdir).mkdir()

            max_intern = max_internal_angle / rad2deg
            min_intern = min_internal_angle / rad2deg

            if overwrite == True:
                Path(f"{workdir}/moirecells.json").unlink(missing_ok=True)
                Path(f"{workdir}/moirecells.cells").unlink(missing_ok=True)

            if Path(f"{workdir}/moirecells.json").exists() == False:
                MatchCells(monos[i], monos[j], workdir, max_coef, tol_theta, store_all, scan_all, sort, max_strain, max_number_of_atoms, min_intern, max_intern)

            


if __name__ == '__main__':
    main()
