import numpy as np
from ase.db import connect
from tqdm import tqdm
from pathlib import Path
from asr.core import command, option


# Angle between two vectors, measured counter-clockwise.
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


# Area of the parallelogram defined by two vectors
def area(v1, v2):
    return abs(v1[0] * v2[1] - v1[1] * v2[0])


class Lincomb:
    def __init__(self, v1, v2, c1, c2):
        self.v1 = v1
        self.v2 = v2
        self.c1 = c1
        self.c2 = c2
        self.comps = [c1 * v1[0] + c2 * v2[0],
                      c1 * v1[1] + c2 * v2[1]]
        self.norm = np.linalg.norm(self.comps)


class Vecpair:
    def __init__(self, lca, lcb):
        self.lca = lca
        self.lcb = lcb
        self.angle = AngleBetween(lca.comps, lcb.comps)

    @property
    def strain(self):
        na = self.lca.norm
        nb = self.lcb.norm
        return max([abs(na - nb) / na * 100,
                    abs(na - nb) / nb * 100])


class Supercell:
    def __init__(self, pair1, pair2):
        self.pair1 = pair1
        self.pair2 = pair2

    @property
    def ang_twist(self):
        return (self.pair1.angle + self.pair2.angle) / 2

    @property
    def ang_int(self):
        ang_a = abs(AngleBetween(self.pair1.lca.comps, self.pair2.lca.comps))
        ang_b = abs(AngleBetween(self.pair1.lcb.comps, self.pair2.lcb.comps))
        return (ang_a + ang_b) / 2

    @property
    def area(self):
        area_a = area(self.pair1.lca.comps, self.pair2.lca.comps)
        area_b = area(self.pair1.lcb.comps, self.pair2.lcb.comps)
        return (area_a + area_b) / 2

    @property
    def strain(self):
        return max(self.pair1.strain, self.pair2.strain)

    def normdiff(self):
        return abs(self.pair1.lca.norm - self.pair2.lca.norm)

    def todict(self):
        rad2deg = 180 / np.pi
        dct = {}
        dct["natoms"] = self.natoms
        dct["m11"] = self.pair1.lca.c1
        dct["m12"] = self.pair1.lca.c2
        dct["m21"] = self.pair2.lca.c1
        dct["m22"] = self.pair2.lca.c2
        dct["n11"] = self.pair1.lcb.c1
        dct["n12"] = self.pair1.lcb.c2
        dct["n21"] = self.pair2.lcb.c1
        dct["n22"] = self.pair2.lcb.c2
        dct["internal_angle"] = self.ang_int * rad2deg
        dct["twist_angle"] = self.ang_twist * rad2deg
        dct["strain"] = self.strain
        return dct


# Returns an iterable range without 0
def modrange(lower, upper):
    fullrange = list(range(lower, upper + 1))
    return fullrange


# Generate all linear combinations for both lattices
def MakeLCs(cell_a, cell_b, scan_all, max_coeff):
    lcs_a = []
    lcs_b = []
    max_a = 0
    max_b = 0
    n_max = 1
    increase = 1

    if scan_all == True:
        m_lower, n_lower = -max_coeff, -n_max
    else:
        m_lower, n_lower = 0, 0

    for m1 in modrange(-max_coeff, max_coeff):
        for m2 in modrange(m_lower, max_coeff):
            if [m1, m2] != [0, 0]:
                lc_a = Lincomb(cell_a[0], cell_a[1], m1, m2)
                lcs_a.append(lc_a)
                if abs(m1) == max_coeff and m2 == max_coeff and lc_a.norm > max_a:
                    max_a = lc_a.norm

    while increase == 1:
        for n1 in modrange(-n_max, n_max):
            for n2 in modrange(n_lower, n_max):
                if [n1, n2] != [0, 0]:
                    lc_b = Lincomb(cell_b[0], cell_b[1], n1, n2)
                    lcs_b.append(lc_b)
                    if abs(n1) == n_max and n2 == n_max and lc_b.norm > max_b:
                        max_b = lc_b.norm
        if max_b > max_a:
            increase = 0
        else:
            lcs_b = []
            n_max += 1
            if scan_all:
                n_lower = - n_max
    return lcs_a, lcs_b


# Find linear combinations with matching moduli (low strain)
def MatchLCs(lcs_a, lcs_b, max_strain):
    matches = []
    for lca in lcs_a:
        for lcb in lcs_b:
            pair = Vecpair(lca, lcb)
            if pair.strain < max_strain:
                matches.append(pair)
    return matches


# Select less strained/smallest/most symmetrical cell if twist angle is the same
def best_duplicate(cell, supercells):
    for sc in supercells:
        if abs(cell.ang_twist - sc.ang_twist) < 1.0e-6:
            if cell.strain < sc.strain - 0.05:
                return supercells.index(sc)
            elif (cell.strain >= sc.strain - 0.05
                  and cell.strain <= sc.strain + 0.05):
                if cell.natoms < sc.natoms:
                    return supercells.index(sc)
                elif cell.natoms == sc.natoms:
                    if cell.normdiff() < sc.normdiff():
                        return supercells.index(sc)
                    else:
                        return -1
                else:
                    return -1
            else:
                return -1
    return -2


# Look for supercells, i.e. LC pairs with matching rotation angle
def FindCells(matches, layer_a, layer_b, tol_theta, min_internal_angle, max_internal_angle, max_number_of_atoms, store_all):
    rad2deg = 180 / np.pi
    supercells = []
    a1, a2 = layer_a.cell[0], layer_a.cell[1]
    b1, b2 = layer_b.cell[0], layer_b.cell[1]
    starting_area_a = area(a1, a2)
    starting_area_b = area(b1, b2)
    nmatches = len(matches)

    print("\nLooking for supercells...")

    angles = []
    for i in tqdm(range(nmatches)):
        for j in range(i + 1, nmatches):
            if abs(matches[i].angle - matches[j].angle) <= tol_theta:
                test_cell = Supercell(matches[i], matches[j])
                ratio_a = test_cell.area / starting_area_a
                ratio_b = test_cell.area / starting_area_b
                natoms = ratio_a * layer_a.natoms + ratio_b * layer_b.natoms
                test_cell.natoms = round(natoms)
                angles.append(matches[i].angle * rad2deg)

                if (test_cell.ang_int >= min_internal_angle
                    and test_cell.ang_int <= max_internal_angle
                        and test_cell.natoms <= max_number_of_atoms):

                    check = best_duplicate(test_cell, supercells)

                    if check == -1:
                        pass

                    elif check == -2 or store_all:
                        supercells.append(test_cell)

                    elif check >= 0 and not store_all:
                        supercells[check] = test_cell
    return supercells


# sort supercells according to number of atoms/strain
def SortResults(supercells, crit):
    if crit == "natoms":
        sorting_list = [cell.natoms for cell in supercells]
    elif crit == "strain":
        sorting_list = [cell.strain for cell in supercells]
    else:
        raise ValueError(
            f"Invalid sorting criteria '{crit}'. Provide either 'natoms' or 'strain'")

    return [supercells[i] for i in np.argsort(sorting_list)]


def SaveJson(supercells, uid_a, uid_b, workdir):
    import json
    results = {'number_of_solutions': len(supercells),
               'uid_a': uid_a,
               'uid_b': uid_b,
               'solutions': {}}
    for i in range(len(supercells)):
        results['solutions'][f"{i}"] = supercells[i].todict()
    file_json = f"{workdir}/moirecells.json"
    with open(file_json, 'w') as f:
        json.dump(results, f, indent=4)


def save_human_readable(supercells, uid_a, uid_b, workdir):
    file_cells = f"{workdir}/moirecells.cells"
    nsol = len(supercells)
    with open(file_cells, "w") as log:
        print("Layer A unique identifier:", file=log)
        print(uid_a, '\n', file=log)
        print("Layer B unique identifier:", file=log)
        print(uid_b, '\n', file=log)
        print("Number of solutions:", file=log)
        print(nsol, '\n', file=log)
        print("--------------------------------------------------- SUPERCELLS --------------------------------------------------------\n", file=log)
        print("{:>3}{:>9}{:>8}{:>5}{:>6}{:>5}{:>8}{:>5}{:>6}{:>5}{:>17}{:>15}{:>11}\n".format("#", "Atoms",
                                                                                                    "m1", "m2", "m1'", "m2'", "n1", "n2", "n1'", "n2'", "Angle(intern)", "Angle(twist)", "Strain(%)"), file=log)
        for i, cell in enumerate(supercells):
            dct = supercells[i].todict()
            print("{:3}".format(i), end='', file=log)
            print("{:8.0f}".format(dct["natoms"]), end='', file=log)
            print("{:>10}".format(dct["m11"]), end='', file=log)
            print("{:>5}".format(dct["m12"]), end='', file=log)
            print("{:>5}".format(dct["m21"]), end='', file=log)
            print("{:>5}".format(dct["m22"]), end='', file=log)
            print("{:>9}".format(dct["n11"]), end='', file=log)
            print("{:>5}".format(dct["n12"]), end='', file=log)
            print("{:>5}".format(dct["n21"]), end='', file=log)
            print("{:>5}".format(dct["n22"]), end='', file=log)
            print("{:>16.6f}".format(dct["internal_angle"]), end='', file=log)
            print("{:>16.6f}".format(dct["twist_angle"]), end='', file=log)
            print("{:>11.4f}\n".format(dct["strain"]), end='', file=log)


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
@option('--max-number-of-atoms', type=float,
        help='Store only supercells with lower number f atoms than the specified one')
@option('--min-internal-angle', type=float,
        help='Lower limit for the supercell internal angle (in degrees)')
@option('--max-internal-angle', type=float,
        help='Upper limit for the supercell internal angle (in degrees)')
@option('--overwrite', type=bool, is_flag=True,
        help='Regenerate directory structure overwriting old files')
@option('--database', type=str,
        help='Path of the .db database file for retrieving structural information')
@option('--uids', type=str,
        help='Path of the file containing the unique ID list of the materials to combine')
@option('--uid-a', type=str)
@option('--uid-b', type=str)
def main(max_coef: int = 10,
         tol_theta: float = 0.05,
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
         database: str = "/home/niflheim/steame/hetero-bilayer-project/databases/c2db.db",
         uids: str = "/home/niflheim/steame/venvs/het-bil/asr/asr/test/moire/tree/uids"):

    if uid_a and uid_b:
        monos = [uid_a, uid_b]
        range_i = [0]
        range_j = [1]
    elif uid_a and not uid_b:
        raise ValueError('Please specify UID for layer B')
    elif uid_b and not uid_a:
        raise ValueError('Please specify UID for layer A')
    else:
        with open(uids, "r") as f:
            monos = [i.split()[0] for i in f.readlines()]
            range_i = range(len(monos))
            range_j = range(i, len(monos))

    db = connect(database)

    for i in range_i:
        for j in range_j:
            layer_a = db.get(uid=monos[i])
            layer_b = db.get(uid=monos[j])
            name_a = layer_a.formula
            name_b = layer_b.formula
            workdir = f"{name_a}-{name_b}"
            dirwork = f"{name_b}-{name_a}"

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

                lcs_a, lcs_b = MakeLCs(cell_a, cell_b, scan_all, max_coef)
                print(
                    f"\nGenerated:\n\t{len(lcs_a):<4} LCs for lattice A\n\t{len(lcs_b):<4} LCs for lattice B\n")

                matches = MatchLCs(lcs_a, lcs_b, max_strain)
                print(f"Obtained {len(matches)} LCs matches")

                cells = FindCells(matches, layer_a, layer_b, tol_theta,
                                  min_intern, max_intern, max_number_of_atoms, store_all)
                print(f"\nFound {len(cells)} supercells!!!\n")

                cells_sorted = SortResults(cells, sort)
                SaveJson(cells_sorted, monos[i], monos[j], workdir)
                save_human_readable(cells_sorted, uid_a, uid_b, workdir)


if __name__ == '__main__':
    main().cli()
