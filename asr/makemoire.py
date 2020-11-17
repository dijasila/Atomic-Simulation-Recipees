import argparse
import numpy as np
from numpy import linalg
from ase import Atoms
from ase.db import connect
from ase.io.jsonio import write_json, read_json
from ase.io import read, write
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from asr.core import command, option




# --------------------------------- FUNCTIONS ---------------------------------- #


def LinComb(m1, m2, v1, v2):
    return [ m1*v1[0] + m2*v2[0], m1*v1[1] + m2*v2[1], v1[2] ]
    


def SumVec(v1, v2):
    if len(v1) == 3 and len(v2) == 3:
        return [ v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2] ]
    if len(v1) == 2 and len(v2) == 2:
        return [ v1[0] + v2[0], v1[1] + v2[1] ]



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



# Counter-clockwise vector rotation
def RotVec(v, th):
    if len(v) == 2:
        rot_matrix = [[np.cos(th), -np.sin(th)],    
                      [np.sin(th), np.cos(th),]]
    if len(v) == 3:
        rot_matrix = [ [np.cos(th), -np.sin(th), 0.0],    
                       [np.sin(th), np.cos(th),  0.0],
                       [0.0       , 0.0        , 1.0] ]
    return np.dot(rot_matrix, v)



# Obtain slope and intercept of the line passing through p1 and p2
def GetMQ(p1, p2):
    if p2[0] == p1[0]:
        return ["vert", p1[0]]
    return [ (p2[1] - p1[1]) / (p2[0] - p1[0]),
              -p1[0] * (p2[1] - p1[1]) / (p2[0] -p1[0]) + p1[1] ]

        

# Scale x and y coordinates of a list of atoms
def ScaleXY(vec, fac_x, fac_y):
    sosia = vec.copy()
    for i in sosia:
        i[0] = i[0] / fac_x
        i[1] = i[1] / fac_y
    return (sosia)



def norm(vec):
    return np.linalg.norm(vec)



# Find a common set of lattice vectors in the bilayer 
# by compressing one layer and stretching the other,
# in a way that minimizes the total stress.
# This approximate, yet accurate method uses only the  
# xx and yy components of the stiffness tensors
def StrainApprox(stif_a, stif_b, pos_a, pos_b):
    xx_ratio_ab = stif_a[0] / stif_b[0]
    xx_ratio_ba = stif_b[0] / stif_a[0]
    yy_ratio_ab = stif_a[1] / stif_b[1]
    yy_ratio_ba = stif_b[1] / stif_a[1]
    dx_a = (pos_b[0] - pos_a[0]) / (1 + xx_ratio_ab)
    dx_b = (pos_a[0] - pos_b[0]) / (1 + xx_ratio_ba)
    dy_a = (pos_b[1] - pos_a[1]) / (1 + yy_ratio_ab)
    dy_b = (pos_a[1] - pos_b[1]) / (1 + yy_ratio_ba)
    x_eq = pos_a[0] + dx_a
    x_eq_test = pos_b[0] + dx_b
    y_eq = pos_a[1] + dy_a
    y_eq_test = pos_b[1] + dy_b
    if abs(x_eq - x_eq_test) < 1.0e-10 and abs(y_eq - y_eq_test) < 1.0e-10:
        print(pos_a)
        print(pos_b)
        print([x_eq, y_eq])
        return [x_eq, y_eq]
    else:
        raise ValueError("Something went wrong...equilibrium cell vectors don't match!")



# Exact method: based on elastic energy minimization
# under the constraint that A1-B1 coordinates must coincide.
# No shear strain/stress is assumed.
def StrainExact(sa, sb, a1, b1, a2, b2, solution):

    # Avoid dividing 0 by 0 in the next step
    if a1[0] < 1.0e-4 and b1[0] < 1.0e-4:
        Rx = 1.0
    else:
        Rx = a1[0] / b1[0]
    if a1[1] < 1.0e-4 and b1[1] < 1.0e-4:
        Ry = 1.0
    else:
        Ry = a1[1] / b1[1]
    
    # Find the equilibrium point S1 for the first vector pair a1, b1
    K = sa[2] + sb[2] * Rx * Ry 
    epsfindmat = np.array([[sa[0] + sb[0] * Rx**2, K],
                           [K, sa[1] + sb[1] * Ry**2]])
    epsfindvec = np.array([sb[0] * Rx * (1 - Rx) + sb[2] * Rx * (1 - Ry),
                           sb[1] * Ry * (1 - Ry) + sb[2] * Ry * (1 - Rx)])
    eps = np.linalg.inv(epsfindmat).dot(epsfindvec)
    epsmat = np.array([[eps[0]+1, 0],
                       [0, eps[1]+1]])
    S1 = np.dot(a1, epsmat)

    # Determine at which point of the difference vector (referred to the shortest
    # vector between a1 and b1) the equilibrium point falls  
    diff1 = abs(norm(a1) - norm(b1))
    diff2 = abs(norm(a2) - norm(b2))
    if norm(a1) < norm(b1):
        diff_scale = (norm(S1) - norm(a1)) / diff1
    else:
        diff_scale = (norm(S1) - norm(b1)) / diff1

    # determine how much the smallest vector between a2 and b2 has to be scaled
    # to meet the equilibrium point S2, then find S2
    increment = diff_scale * diff2
    if norm(a2) < norm(b2):
        ang = AngleBetween([1.0, 0.0], a2) 
        S2norm = norm(a2) + increment
    else:
        ang = AngleBetween([1.0, 0.0], b2) 
        S2norm = norm(b2) + increment
    S2 = [S2norm * np.cos(ang), S2norm * np.sin(ang)]

    if abs(S1[0]) < 1.0e-10:
        S1[0] = 0.0
    if abs(S1[1]) < 1.0e-10:
        S1[1] = 0.0
    if abs(S2[0]) < 1.0e-10:
        S2[0] = 0.0
    if abs(S2[1]) < 1.0e-10:
        S2[1] = 0.0

    errors = []
    # Debugging: new vector components should lay in between the starting ones
    error = 0
    for i, j, k, l, m, n in zip(S1, a1, b1, S2, a2, b2):
        if abs(i-j) > abs(j-k): 
            error = 1
        if abs(l-m) > abs(m-n):
            error = 2

    if error != 0:
        with open("errors.log", "a") as f:
            print(str(datetime.now()), file=f)
            print(f"Lattice match error found for supercell {solution}. Used approximate matching method", "\n", file=f)
            print(f"A1: {a1}", file=f)
            print(f"B1: {b1}", file=f)
            print(f"S1: {S1}", file=f)
            print(f"A2: {a2}", file=f)
            print(f"B2: {b2}", file=f)
            print(f"S2: {S2}", file=f)
            print("\n", file=f)
        return ["err", 0]

    else:
        return np.array([S1, S2])



def MakeFormula(vec):
    symbol = vec[0]
    count = 1
    form = []
    for i in range(1, len(vec)):
        if vec[i] != vec[i-1]:
            if count > 1:
                form.append("{0}{1}".format(symbol, count))
            else:
                form.append("{0}".format(symbol))
            symbol = vec[i]
            count = 1
        else:
            count += 1
    if count > 1:
        form.append("{0}{1}".format(symbol, count))
    else:
        form.append("{0}".format(symbol))
    return "".join(form)



rad2deg = 180 / np.pi



def MakeCell(cells_file, solution, tol, stress_opt_method, database, overwrite):


    # ----------------------------- CELL DEFINITION -------------------------------- #
    
    with open(cells_file, 'r') as f:
        dct = read_json(cells_file) 
    uid_a = dct["uid_a"]
    uid_b = dct["uid_b"]
    sc_info = dct["solutions"][solution]
        
    # Get atoms objects for the two layers from C2DB
    db = connect(database)
    comp_a = db.get(uid=uid_a)
    comp_b = db.get(uid=uid_b)
    
    layer_a = { "name"            : comp_a.formula,
                "v1"              : comp_a.cell[0],         
                "v2"              : comp_a.cell[1],         
                "v3"              : comp_a.cell[2],         
                "positions"       : comp_a.positions.copy(),         
                "symbols"         : comp_a.symbols,         
                "numbers"         : comp_a.numbers,         
                "N-atoms-start"   : comp_a.natoms,         
                "c1-1"            : sc_info["indexes_a"][0],         
                "c2-1"            : sc_info["indexes_a"][1],
                "c1-2"            : sc_info["indexes_a"][2],         
                "c2-2"            : sc_info["indexes_a"][3] }
    
    layer_b = { "name"            : comp_b.formula,
                "v1"              : comp_b.cell[0],         
                "v2"              : comp_b.cell[1],         
                "v3"              : comp_b.cell[2],         
                "positions"       : comp_b.positions,         
                "symbols"         : comp_b.symbols,         
                "numbers"         : comp_b.numbers,         
                "N-atoms-start"   : comp_b.natoms,
                "c1-1"            : sc_info["indexes_b"][0],         
                "c2-1"            : sc_info["indexes_b"][1],
                "c1-2"            : sc_info["indexes_b"][2],         
                "c2-2"            : sc_info["indexes_b"][3] }
    
    rot_angle = sc_info["twist_angle"]
    strain = sc_info["max_strain"]
    
    
    
    # ----------------------- ADJUST STARTING VERTICAL OFFSET ---------------------- # 
    
    
    # Assign the interlayer distances in the bulk phase starting from database (if available)
    if comp_a.has_bulk_dist == True:
        bulkdist_a = comp_a.bulk_dist
        bulkdist_b = comp_b.bulk_dist
        exp_avg_dist = (bulkdist_a + bulkdist_b) / 2
    else:
        exp_avg_dist = 3.0
    
    # Do stuff with the starting coordinates to setup vacuum thickness
    zvals_a = [i[2] for i in layer_a["positions"]]
    zvals_b = [j[2] for j in layer_b["positions"]]
    
    thickness_b = max(zvals_b) - min(zvals_b)
    starting_offset = min(zvals_b) - min(zvals_a)
    
    exp_avg_dist = (bulkdist_a + bulkdist_b) / 2
    vert_shift_a = starting_offset + thickness_b + exp_avg_dist
    
    pos_a = layer_a["positions"]
    
    for l in pos_a:
        l[2] += vert_shift_a
    zvals_a_new = [ l[2] for l in pos_a]
    zvals_new = zvals_b + zvals_a_new
    
    # Set up 15 Ang vacuum
    z_const_supercell = 15 + (max(zvals_new) - min(zvals_new))
    
    
    
    # ----------------------------- GENERATE NEW ATOMS ----------------------------- # 
    
    
    for lyr in layer_a, layer_b:
    
        # Define the corners of the new unit cell, for shifting purposes
        c0 = [0, 0, 0]
        c1 = LinComb(lyr["c1-1"], lyr["c2-1"], lyr["v1"], lyr["v2"])
        c2 = LinComb(lyr["c1-2"], lyr["c2-2"], lyr["v1"], lyr["v2"])
        c3 = SumVec(c1, c2)
        corners = np.array([c0, c1, c2, c3])
        lyr["R1"] = c1
        lyr["R2"] = c2
        
        old_surface = abs(lyr["v1"][0] * lyr["v2"][1] - lyr["v1"][1] * lyr["v2"][0])
        new_surface = abs(c2[0] * c3[1] - c2[1] * c3[0])
    
        surf_ratio = int(round(new_surface / old_surface, 0))
        lyr["N-atoms"] = surf_ratio * len(lyr["symbols"])
        
        # Sort corners by x value
        corners_sort_x = corners[:, 0].argsort()
        corners = corners[corners_sort_x]
        
        # Store the x value order
        xvals = [ i[0] for i in corners ]
        
        # Swap corner 2 with corner 1 if the angle between 0-1 and 0-2 edges is positive (i.e. 0-1 lays below 0-2)
        edge_01 = [ corners[1,0] - corners[0,0], 
                    corners[1,1] - corners[0,1] ] 
        edge_02 = [ corners[2,0] - corners[0,0], 
                    corners[2,1] - corners[0,1] ] 
        if AngleBetween(edge_01, edge_02) > 0: 
            corners[[1, 2]] = corners[[2, 1]]
    
        lyr["corners"] = corners
    
        # Store slope and intercept of the cell borders
        # e.g. mq_01: slope and intercept of the border connecting corners 0 and 1
        mq_01 = GetMQ(corners[0], corners[1])
        mq_02 = GetMQ(corners[0], corners[2])
        mq_13 = GetMQ(corners[1], corners[3])
        mq_23 = GetMQ(corners[2], corners[3])
    
        # If the lattice has a base vector - and, consequently, two borders - 
        # parallel to the y axis, we only need the other two borders.
        if mq_01[0] == "vert" or mq_02[0] == "vert" or mq_13[0] == "vert" or mq_23[0] == "vert": 
            vertical = 1
        else:
            vertical = 0
        
        # Determine the coefficient range in which to find new atoms
        coef_max_v1 = max(lyr["c1-1"], lyr["c1-2"], lyr["c1-1"] + lyr["c1-2"], 0)
        coef_min_v1 = min(lyr["c1-1"], lyr["c1-2"], lyr["c1-1"] + lyr["c1-2"], 0)
        coef_max_v2 = max(lyr["c2-1"], lyr["c2-2"], lyr["c2-1"] + lyr["c2-2"], 0)
        coef_min_v2 = min(lyr["c2-1"], lyr["c2-2"], lyr["c2-1"] + lyr["c2-2"], 0)
    
    
    
        ### GENERATE ATOM REPLICAS AND GROUP THEM ### 
    
        group_1 = []
        group_2 = []
        group_3 = []
        group_1_symbols = []
        group_2_symbols = []
        group_3_symbols = []
        group_1_numbers = []
        group_2_numbers = []
        group_3_numbers = []
        
        
        # For each atom in th unit cell...
        for atom, symbol, number in zip(lyr["positions"], lyr["symbols"], lyr["numbers"]):
    
            # ...for each LC inside the coefficient ranges defined above...
            for coef1 in range(coef_min_v1, coef_max_v1 + 1):
                for coef2 in range(coef_min_v2, coef_max_v2 + 1):
    
                    # ...shift that atom position (skipping the atom symbol) by that LC, and include it in a temporary list
                    shift = LinComb(coef1, coef2, lyr["v1"], lyr["v2"])
                    atom_new = SumVec(atom, shift)
    
                    # Assign the new atom to a group based on its x value
                    # Tolerance is necessary in order to avoid counting the corners/borders multiple times
                    # and to take into account the finite precision of the algorithm
                    if atom_new[0] >= xvals[0] - tol and atom_new[0] < xvals[1] - tol:
                        group_1.append(atom_new)
                        group_1_symbols.append(symbol)
                        group_1_numbers.append(number)
                    if atom_new[0] >= xvals[1] - tol and atom_new[0] < xvals[2] - tol:
                        group_2.append(atom_new)
                        group_2_symbols.append(symbol)
                        group_2_numbers.append(number)
                    if atom_new[0] >= xvals[2] - tol and atom_new[0] < xvals[3] + tol:
                        group_3.append(atom_new)
                        group_3_symbols.append(symbol)
                        group_3_numbers.append(number)
     
    
    
        ### SEARCH FOR NEW ATOMS INSIDE THE SUPERCELL ###
    
        # Divide the x axis in three portions based on the corners x values. For each portion
        # define the upper and lower limits, given by the equations of the lines connecting
        # the borders which lie in the specified x range.
        #
        # N.B: The equal signs in the following if statements are very important!!! 
        #      I always want to keep the points (if any) on the borders 0-1 and 0-2
        #      and discard the equivalent ones on borders 1-3 and 2-3
    
        skip = 0
        lyr["supercell"] = []
        lyr["supercell_labels"] = []
        lyr["supercell_numbers"] = []
        lyr["kept"] = []
        lyr["skipped"] = []
    
    
        for atom_g1, label_g1, num_g1 in zip(group_1, group_1_symbols, group_1_numbers):
        
            upper_limit = mq_01[0] * atom_g1[0] + mq_01[1] 
            lower_limit = mq_02[0] * atom_g1[0] + mq_02[1] 
            # Include the atoms located on the borders 0-1 and 0-2
            if atom_g1[1] >= lower_limit - tol and atom_g1[1] <= upper_limit + tol:
                lyr["supercell"].append(atom_g1)
                lyr["supercell_labels"].append(label_g1)
                lyr["supercell_numbers"].append(num_g1)
                lyr["kept"].append([atom_g1[0], atom_g1[1]])
            else: 
                lyr["skipped"].append([atom_g1[0], atom_g1[1]])
        
    
    
        for atom_g2, label_g2, num_g2 in zip(group_2, group_2_symbols, group_2_numbers):
        
            # If the point coincides with one of the corners, skip it
            if abs(atom_g2[0] - corners[1][0]) < tol and abs(atom_g2[1] - corners[1][1]) < tol:
                skip = 1
                lyr["skipped"].append([atom_g2[0], atom_g2[1]])
            elif abs(atom_g2[0] - corners[2][0]) < tol and abs(atom_g2[1] - corners[2][1]) < tol:
                skip = 1
                lyr["skipped"].append([atom_g2[0], atom_g2[1]])
        
            # Case in which two borders are parallel to y axis
            if vertical == 1: 
                # Set skip to 1 only to avoid the next steps
                skip = 1
                upper_limit = mq_13[0] * atom_g2[0] + mq_13[1]
                lower_limit = mq_02[0] * atom_g2[0] + mq_02[1] 
                if atom_g2[1] >= lower_limit - tol and atom_g2[1] < upper_limit - tol:
                    lyr["supercell"].append(atom_g2)
                    lyr["supercell_labels"].append(label_g2)
                    lyr["supercell_numbers"].append(num_g2)
                    lyr["kept"].append([atom_g2[0], atom_g2[1]])
                else: 
                    lyr["skipped"].append([atom_g2[0], atom_g2[1]])
            
    
            if skip == 0:
                if corners[1][0] < corners[2][0]: 
                    upper_limit = mq_13[0] * atom_g2[0] + mq_13[1] 
                    lower_limit = mq_02[0] * atom_g2[0] + mq_02[1] 
                    # Include the atoms on the border 0-2, exclude the ones on 1-3
                    if atom_g2[1] >= lower_limit - tol and atom_g2[1] < upper_limit - tol:
                        lyr["supercell"].append(atom_g2)
                        lyr["supercell_labels"].append(label_g2)
                        lyr["supercell_numbers"].append(num_g2)
                        lyr["kept"].append([atom_g2[0], atom_g2[1]])
                    else: 
                        lyr["skipped"].append([atom_g2[0], atom_g2[1]])
         
                if corners[1][0] > corners[2][0]: 
                    upper_limit = mq_01[0] * atom_g2[0] + mq_01[1] 
                    lower_limit = mq_23[0] * atom_g2[0] + mq_23[1] 
                    # Include the atoms on the border 0-1, exclude the ones on 2-3
                    if atom_g2[1] > lower_limit + tol and atom_g2[1] <= upper_limit + tol:
                        lyr["supercell"].append(atom_g2)
                        lyr["supercell_labels"].append(label_g2)
                        lyr["supercell_numbers"].append(num_g2)
                        lyr["kept"].append([atom_g2[0], atom_g2[1]])
                    else: 
                        lyr["skipped"].append([atom_g2[0], atom_g2[1]])
            skip = 0
        
    
    
        for atom_g3, label_g3, num_g3 in zip(group_3, group_3_symbols, group_3_numbers):
        
            if abs(atom_g3[0] - corners[3][0]) < tol and abs(atom_g3[1] - corners[3][1]) < tol:
                skip = 1
                lyr["skipped"].append([atom_g3[0], atom_g3[1]])

            if vertical == 1:
                skip = 1
                lyr["skipped"].append([atom_g3[0], atom_g3[1]])
        
            if skip == 0:
                upper_limit = mq_13[0] * atom_g3[0] + mq_13[1] 
                lower_limit = mq_23[0] * atom_g3[0] + mq_23[1] 
                if atom_g3[1] > lower_limit + tol and atom_g3[1] < upper_limit - tol:
                    lyr["supercell"].append(atom_g3)
                    lyr["supercell_labels"].append(label_g3)
                    lyr["supercell_numbers"].append(num_g3)
                    lyr["kept"].append([atom_g3[0], atom_g3[1]])
                else: 
                    lyr["skipped"].append([atom_g3[0], atom_g3[1]])
            skip = 0
    
    
        # Notice that in the "vertical" case we always have
        #  - empty group 1
        #  - group 3 containing only the points located on border 2-3, which we don't need.
    
        lyr["kept"] = np.array(lyr["kept"])
        lyr["skipped"] = np.array(lyr["skipped"])
    
    
    layer_a["layerlevel"] = []
    layer_b["layerlevel"] = []
    for atom_a in layer_a["supercell"]:
        layer_a["layerlevel"].append(1)
    for atom_b in layer_b["supercell"]:
        layer_b["layerlevel"].append(0)
    
    
    ### ROTATE LATTICE A ###
    
    supercell_rot = []
    for atom in layer_a["supercell"]:
        supercell_rot.append(RotVec(atom, rot_angle / rad2deg))
    layer_a["supercell"] = np.array(supercell_rot)
    layer_a["R1"] = RotVec(layer_a["R1"], rot_angle / rad2deg)
    layer_a["R2"] = RotVec(layer_a["R2"], rot_angle / rad2deg)
    
    
    
    # ----------------------------- APPLY STRAIN ----------------------------- # 
    
    
    # If the cell mismatch is nonzero, adjust cell lengths
    if strain > 1.0e-06:
    
        # Get xx and yy stiffness tensor components from C2DB
        stiff_a = [comp_a.c_11, comp_a.c_22, comp_a.c_12]
        stiff_b = [comp_b.c_11, comp_b.c_22, comp_b.c_12]
        
        # Define the starting coordinates for all four vectors
        coords_a1 = [layer_a["R1"][0], layer_a["R1"][1]]
        coords_a2 = [layer_a["R2"][0], layer_a["R2"][1]]
        coords_b1 = [layer_b["R1"][0], layer_b["R1"][1]]
        coords_b2 = [layer_b["R2"][0], layer_b["R2"][1]]
        
        if stress_opt_method == "approx":
            S1 = StrainApprox(stiff_a, stiff_b, coords_a1, coords_b1)
            S2 = StrainApprox(stiff_a, stiff_b, coords_a2, coords_b2)

        else: 
            S = StrainExact(stiff_a, stiff_b, coords_a1, coords_b1, coords_a2, coords_b2, solution)
            # If errors are found, use approximate method
            if S[0] == "err":
                S1 = StrainApprox(stiff_a, stiff_b, coords_a1, coords_b1)
                S2 = StrainApprox(stiff_a, stiff_b, coords_a2, coords_b2)
            else: 
                S1 = S[0]
                S2 = S[1]
    
    # Otherwise, just use the lattice vectors of one of the two layers
    else:
        S1 = layer_b["R1"]
        S2 = layer_b["R2"]
        
    # Obtain atoms fractional coordinates using the old vectors
    R1_a_norm = np.linalg.norm(layer_a["R1"])
    R2_a_norm = np.linalg.norm(layer_a["R2"])
    R1_b_norm = np.linalg.norm(layer_b["R1"])
    R2_b_norm = np.linalg.norm(layer_b["R2"])
    
    a_scaled = ScaleXY(layer_a["supercell"], R1_a_norm, R2_a_norm)
    b_scaled = ScaleXY(layer_b["supercell"], R1_b_norm, R2_b_norm)
    
    # Rescale coordinates with respect to the new vectors. 
    # Also if the starting vectors match, just in case...
    S1_norm = np.linalg.norm(S1)
    S2_norm = np.linalg.norm(S2)
    A_FINAL = ScaleXY(a_scaled, 1/S1_norm, 1/S2_norm)
    B_FINAL = ScaleXY(b_scaled, 1/S1_norm, 1/S2_norm)
    
    
    
    # ----------------------- GENERATE AND TEST SUPERCELL ----------------------- # 
    
    
    final_supercell = np.concatenate((A_FINAL, B_FINAL), axis = 0)
    final_supercell_lbl = np.concatenate((layer_a["supercell_labels"], layer_b["supercell_labels"]), axis = 0)
    final_supercell_num = np.concatenate((layer_a["supercell_numbers"], layer_b["supercell_numbers"]), axis = 0)
    supercell_formula = MakeFormula(final_supercell_lbl)
    layerlevels = layer_a["layerlevel"] + layer_b["layerlevel"]
    
    total_atoms = len(final_supercell)
    total_atoms_expect = layer_a["N-atoms"] + layer_b["N-atoms"]
    
    if total_atoms_expect != total_atoms:
        print("\nSomething went wrong... :(\nExpected number of atoms: {0}\nActual number of atoms:   {1}\n\nTry to reduce the tolerance below 1e-10 or, if it doesn't work, increase it.\n".format(total_atoms_expect, total_atoms))
        raise ValueError()
    
    
    
    # ------------------------- OUTPUT SUPERCELL INFO IN .JSON AND .XYZ ---------------------------- #
    
    
    # Store supercell info in a dict convertible into an atoms object
    supercell = {}
    supercell["numbers"] = final_supercell_num
    supercell["cell"] = [[S1[0], S1[1], 0.0], 
                         [S2[0], S2[1], 0.0], 
                         [layer_b["v3"][0], layer_b["v3"][1], z_const_supercell]]
    supercell["positions"] = final_supercell
    supercell["pbc"] = [True, True, False]
    supercell["info"] = {}
    supercell["info"]["layerlevels"] = layerlevels
    supercell["info"]["strain"] = strain
    
    dirname = f"{total_atoms}_{rot_angle:.1f}_{strain:.2f}"
    if Path(dirname).exists() == False: 
        Path(dirname).mkdir()
    
    file_json = f"{dirname}/unrelaxed.json"
    file_xyz = f"{dirname}/unrelaxed.xyz"

    atoms = Atoms.fromdict(supercell)
    atoms.set_tags(layerlevels)

    if Path(file_json).exists() == False or overwrite == True:
        write(file_json, atoms)
    else: 
        print(f"File {file_json} already exists. Pass the --overwrite option if you wish to replace it")

    if Path(file_xyz).exists() == False or overwrite == True:
        write(file_xyz, atoms)
    else: 
        print(f"File {file_xyz} already exists. Pass the --overwrite option if you wish to replace it")


@command('asr.makemoire',
         requires=['moirecells.json'])

@option('--cells-file', type=str, 
        help='Path of the json file containing the supercell list')
@option('--solution', type=int, 
        help='For single supercell generation: specify the index of the supercell to generate, as found in moirecells.cells or moirecells.json')
@option('--tol', type=float)
@option('--stress-opt-method', type=str)
@option('--database', type=str, 
        help="Path to the database file")
@option('--overwrite', type=bool) 

def main(cells_file: str="moirecells.json", solution: int=-1, tol: float=1.0e-10, stress_opt_method: str="exact", database : str='/home/niflheim/steame/hetero-bilayer-project/databases/gw-bulk.db', overwrite: bool = False):

    with open(cells_file, "r") as f:
        dct = read_json(cells_file) 
    nsol = dct["number_of_solutions"]

    if solution >= 0:
        MakeCell(cells_file, solution, tol, stress_opt_method, database, overwrite) 
    else:   
        for sol in tqdm(range(nsol)):
            MakeCell(cells_file, sol, tol, stress_opt_method, database, overwrite)



if __name__ == '__main__':
    main()
