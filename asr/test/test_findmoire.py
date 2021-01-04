import numpy as np
from ase.db import connect
from tqdm import tqdm
from pathlib import Path
from asr.core import command, option


# Angle between v1 and v2, measured counter-clockwise.
# If angle is > 0, you have to rotate v1 counter-clockwise to match v2, and vv.sa
def AngleBetween(v1, v2):
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


# Obtain max relative difference between two vectors
def GetMaxStrain(mod_a, mod_b):
    return max([abs(mod_a - mod_b) / mod_a * 100,
                abs(mod_a - mod_b) / mod_b * 100])


# Return absolute value of the angle between two vectors, between 0° and 180°
def AngIntern(v1, v2):
    val_angle = AngleBetween(v1, v2)
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


def ScanLCs(cell_a, cell_b, scan_all, max_coeff):
    a1 = cell_a[0, 0:2]
    a2 = cell_a[1, 0:2]
    b1 = cell_b[0, 0:2]
    b2 = cell_b[1, 0:2]
    scans = {
             "vectors_a": [],
             "vectors_b": [],
             "moduli_a" : [],
             "moduli_b" : [],
             "indexes_a": [],
             "indexes_b": [],
            }
    max_a = 0
    max_b = 0 
    n_max = 1       
    increase = 1
    # Scan either the upper half of the XY plane or all of it.
    if scan_all == True:
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
                scans["vectors_a"].append(vec)
                scans["moduli_a"].append(norm(vec))
                scans["indexes_a"].append([m1, m2])
                # Store the maximum modulus values in order to set the number of LCs for lattice B
                if abs(m1) == max_coeff and m2 == max_coeff and norm(vec) > max_a:
                    max_a = norm(vec)
    # Generate LCs for lattice B inside shells with increasing n1,n2 values
    while increase == 1:
        for n1 in range( n_max, - n_max - 1, -1 ):
            for n2 in range( n_lower, n_max + 1 ):
                if [n1, n2] != [0, 0]:
                    vec = lc_vec(n1, n2, b1, b2)
                    scans["vectors_b"].append(vec)
                    scans["moduli_b"].append(norm(vec))
                    scans["indexes_b"].append([n1, n2])
                    if abs(n1) == n_max and n2 == n_max and norm(vec) > max_b:
                        max_b = norm(vec)
        # Once a shell is completed, compare the max moduli for lattices A and B.
        # Stop when the ones for lattice B are larger.
        if max_b > max_a: 
            increase = 0
        else:
            scans["vectors_b"], scans["moduli_b"], scans["indexes_b"] = [], [], []
            n_max += 1
            if scan_all == True:
                n_lower = - n_max
    scans["nvecs_a"] = len(scans["vectors_a"])
    scans["nvecs_b"] = len(scans["vectors_b"])
    return scans


# Store the LCs in lattice A and B with matching moduli
def MatchLCs(scans, max_strain):
    matches = {
               "vectors_a": [],
               "vectors_b": [],
               "indexes_a": [],
               "indexes_b": [],
               "angles"   : [],
               "strains"  : [],
               "nmatches" : 0
              }
    range_a = range(scans["nvecs_a"])
    range_b = range(scans["nvecs_b"])
    for i in range_a:
        for j in range_b:
            strain_ij = GetMaxStrain(scans["moduli_a"][i], scans["moduli_b"][j])
            if strain_ij < max_strain:
                angle_ij = AngleBetween(scans["vectors_a"][i],
                                         scans["vectors_b"][j])
                matches["vectors_a"].append(scans["vectors_a"][i])
                matches["vectors_b"].append(scans["vectors_b"][j])
                matches["indexes_a"].append(scans["indexes_a"][i])
                matches["indexes_b"].append(scans["indexes_b"][j])
                matches["angles"].append(angle_ij)
                matches["strains"].append(strain_ij)
                matches["nmatches"] += 1
    return matches


def FindCells(matches, layer_a, layer_b, tol_theta, min_internal_angle, max_internal_angle): 
    nmatches = matches["nmatches"]
    cells = {
             "a1"        : [],
             "a2"        : [],
             "b1"        : [],
             "b2"        : [],
             "a1_coeffs" : [],
             "a2_coeffs" : [],
             "b1_coeffs" : [],
             "b2_coeffs" : [],
             "indexes_a" : [],
             "indexes_b" : [],
             "ang_twist" : [],
             "ang_int"   : [],
             "strains"   : [],
             "ratios_a"  : [],
             "ratios_b"  : [],
             "areas"     : [],
             "natoms"    : [],
             "ncells"    : 0
            }

    print("\nGenerating supercells...\n")
    
    match_num = matches["nmatches"]
    a1, a2 = layer_a.cell[0, 0:2], layer_a.cell[1, 0:2]
    b1, b2 = layer_b.cell[0, 0:2], layer_b.cell[1, 0:2]
    starting_area_a = area(a1, a2)
    starting_area_b = area(b1, b2)

    for i in tqdm(range(match_num)):
        for j in range(i + 1, match_num):
            # For commensurate lattices, we will always obtain all the multiples of the  
            # superposition of the unit cells. Let's discard them, and keep only the first superposition.
            if ([match["indexes_a"][i], match["indexes_b"][i]] == [[1, 0], [1, 0]]  or
                match["indexes_a"][i] != match["indexes_b"][i] ):

                ratio_a = area(matches["vectors_a"][i], matches["vectors_a"][j]) / starting_area_a
                ratio_b = area(matches["vectors_b"][i], matches["vectors_b"][j]) / starting_area_b
                natom = ratio_a * layer_a.natoms + ratio_b * layer_b.natoms
                test_intern = AngIntern(matches["vectors_a"][i], matches["vectors_a"][j])
                test_area = area(matches["vectors_a"][i], matches["vectors_a"][j])
                test_twist = (matches["angles"][i] + matches["angles"][j]) / 2
                strain_ij = max(matches["strains"][i], matches["strains"][j])

                if (abs(matches["angles"][i] - matches["angles"][j]) <= tol_theta and 
                    test_intern >= min_internal_angle and 
                    test_intern <= max_internal_angle and 
                    natom <= max_number_of_atoms ):

                    check = best_duplicate(test_twist, cells["ang_twist"], 
                                           strain_ij, cells["strains"], 
                                           test_area, cells["areas"], 
                                           test_intern, cells["ang_int"])
    
                    # Save new supercells
                    if check == [1, -1] or store_all == True:
                        cells["a1"].append(matches["vectors_a"][i])
                        cells["a2"].append(matches["vectors_a"][j])
                        cells["b1"].append(matches["vectors_b"][i])
                        cells["b2"].append(matches["vectors_b"][j])
                        cells["a1_coeffs"].append(matches["indexes_a"][i]) 
                        cells["a2_coeffs"].append(matches["indexes_a"][j])
                        cells["b1_coeffs"].append(matches["indexes_b"][i])
                        cells["b2_coeffs"].append(matches["indexes_b"][j])
                        cells["ang_twist"].append(test_twist)
                        cells["ang_int"].append(test_intern)
                        cells["areas"].append(test_area)
                        cells["ratios_a"].append(ratio_a)
                        cells["ratios_b"].append(ratio_b)
                        cells["natoms"].append(natom)
                        cells["strains"].append(strain_ij)
    
                    # Replace equivalent supercells if one with a smaller rotation angle is found.
                    if check[0] == -1 and store_all == False:
                        cells["a1"][check[1]] = matches["vectors_a"][i]
                        cells["a2"][check[1]] = matches["vectors_a"][j]
                        cells["b1"][check[1]] = matches["vectors_b"][i]
                        cells["b2"][check[1]] = matches["vectors_b"][j]
                        cells["a1_coeffs"][check[1]] = matches["indexes_a"][i]
                        cells["a2_coeffs"][check[1]] = matches["indexes_a"][j]
                        cells["b1_coeffs"][check[1]] = matches["indexes_b"][i]
                        cells["b2_coeffs"][check[1]] = matches["indexes_b"][j]
                        cells["ang_twist"][check[1]] = test_twist
                        cells["ang_int"][check[1]] = test_intern
                        cells["areas"][check[1]] = test_area
                        cells["ratios_a"][check[1]] = ratio_a
                        cells["ratios_b"][check[1]] = ratio_b 
                        cells["natoms"].[check[1]]  = natom
                        cells["strains"][check[1]] = strain_ij
    return cells


# Sort "vec" based on the values contained in "sort_vec"
def CustomSort(vec, sort_vec):
    return [ vec[i] for i in np.argsort(sort_vec) ]


def SortResults(cells, crit):
    sorted_result = {}
    if crit == "strain":
        sortvec = cells["strains"]
    elif crit == "natoms":
        sortvec = cells["natoms"]
    for key in cells:
        if type(cells[key]) == list:
            sorted_result[key] = CustomSort(cells[key], sortvec)
            

@command('asr.findmoire')

@option('--max-coef', type=int, 
        help='Max coefficient for linear combinations of the starting vectors')
@option('--tol-theta', type=float, 
        help='Tolerance over rotation angle difference between matching vector pairs')
@option('--store-all', type=bool, 
        help='True: store all the possible matches. False: store only unique supercell')
@option('--scan-all', type=bool, 
        help='True: scan linear combinations in all the XY plane. False: scan only the upper half')
@option('--sort', type=str, 
        help='Sort results by number of atoms or max strain')
@option('--max-strain', type=float, 
        help='Store only supercells with max (percent) strain lower than the specified one')
@option('--max-natoms', type=float, 
        help='Store only supercells with lower number f atoms than the specified one')
@option('--min-internal-angle', type=float, 
        help='Lower limit for the supercell internal angle (in degrees)')
@option('--max-internal-angle', type=float, 
        help='Upper limit for the supercell internal angle (in degrees)')
@option('--overwrite', type=bool, 
        help='True: Regenerate directory structure overwriting old files; False: generate results only for new entries')
@option('--database', type=str, 
        help='Path of the .db database file for retrieving structural information')
@option('--uids', type=str, 
        help='Path of the file containing the unique ID list of the materials to combine')
@option('--uid-a', type=str)
@option('--uid-b', type=str)

def main(max_coef: int = 10,
         tol_theta: float = 0.05
         store_all: bool = False, 
         scan_all: bool = False, 
         sort: str = "natoms", 
         max_strain: float = 1.0, 
         max_number_of_atoms: int = 300, 
         min_internal_angle: float = 30.0, 
         max_internal_angle: float = 150.0, 
         overwrite: str = False, 
         uid_a: str = None, 
         uid_b: str = None, 
         database: str = "/home/niflheim/steame/hetero-bilayer-project/databases/gw-bulk.db", 
         uids: str = "/home/niflheim/steame/venvs/hetero-bilayer-new/venv/asr/asr/test/moire/tree/uids"):

    if uid_a is not None and uid_b is not None:
        uids = [uid_a, uid_b]
        range_i = [0]
        range_j = [1]
    else: 
        with open(uids, "r") as f:
            uids = [ i.split()[0] for i in f.readlines() ]
            range_i = range(len(monos))
            range_j = range(i, len(monos))

    db = connect(database)

    for i in range_i:
        for j in range_j:
            layer_a = db.get(uid=uids[i])
            layer_b = db.get(uid=uids[j])
            name_a = layer_a.formula
            name_b = layer_b.formula
            workdir = f"{name_a}-{name_b}"
            dirwork = f"{name_b}-{name_a}"
            if uids[i] == uids[j]:
                same = True

            # Generate directory and results for the current bilayer if they don't exist
            # or "overwrite" option is passed"
            if Path(workdir).exists() == False and Path(dirwork).exists() == False:
                Path(workdir).mkdir()

            if overwrite == True:
                try:
                    Path(f"{workdir}/moirecells.json").unlink()
                    Path(f"{workdir}/moirecells.cells").unlink()
                except:
                    pass

            if Path(f"{workdir}/moirecells.json").exists() == False:
                rad2deg = 180 / np.pi
                max_intern = max_internal_angle / rad2deg
                min_intern = min_internal_angle / rad2deg
                tol_theta = tol_theta / rad2deg
                max_intern = max_internal_angle / rad2deg
                min_intern = min_internal_angle / rad2deg
                cell_a = layer_a.cell
                cell_b = layer_b.cell

                scans = ScanLCs(cell_a, cell_b, scan_all, max_coef)
                matches = MatchLCs(scans, max_strain)
                cells = FindCells(matches, layer_a, layer_b, tol_theta, min_internal_angle, max_internal_angle)
                cells_sorted = SortResults(cells, sort)


if __name__ == '__main__':
    main()
