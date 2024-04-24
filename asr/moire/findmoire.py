import numpy as np
from ase.db import connect
from ase.build import make_supercell
from ase.io.jsonio import read_json, write_json
from tqdm import tqdm
from pathlib import Path
from pandas import DataFrame
from asr.core import command, option
from typing import Union


class Lincomb:
    def __init__(self, atoms, coeffs):
        self.coeffs = coeffs
        self._atoms = atoms
        self.comps = atoms.cell[0] * coeffs[0] + atoms.cell[1] * coeffs[1]
        self.norm = np.linalg.norm(self.comps)


class Vecpair:
    ''' Pair of vectors with matching moduli:
        one from lattice A, the other from lattice B.
    '''

    def __init__(self, lca, lcb, strain):
        self.lca = lca
        self.lcb = lcb
        self.strain = strain
        self.angle = angle_between(lca.comps, lcb.comps)


def get_monolayers(sc):
    T_a = [[sc.pair1.lca.coeffs[0], sc.pair1.lca.coeffs[1], 0],
           [sc.pair2.lca.coeffs[0], sc.pair2.lca.coeffs[1], 0],
           [0, 0, 1]]
    T_b = [[sc.pair1.lcb.coeffs[0], sc.pair1.lcb.coeffs[1], 0],
           [sc.pair2.lcb.coeffs[0], sc.pair2.lcb.coeffs[1], 0],
           [0, 0, 1]]
    layer_a = make_supercell(sc.pair1.lca._atoms, T_a)
    layer_b = make_supercell(sc.pair1.lcb._atoms, T_b)


class Supercell:
    def __init__(self, pair1, pair2, const):
        self.pair1 = pair1
        self.pair2 = pair2
        self._const = const

    def get_twist_angle(self):
        return (self.pair1.angle + self.pair2.angle) / 2

    def get_internal_angle(self):
        ang_a = abs(angle_between(self.pair1.lca.comps, self.pair2.lca.comps))
        ang_b = abs(angle_between(self.pair1.lcb.comps, self.pair2.lcb.comps))
        return (ang_a + ang_b) / 2

    def get_natoms(self):
        area_a = area(self.pair1.lca.comps, self.pair2.lca.comps)
        area_b = area(self.pair1.lcb.comps, self.pair2.lcb.comps)
        return int((self._const * (area_a + area_b) / 2).round(0))

    def get_norm_ratio(self):
        return self.pair1.lca.norm / self.pair2.lca.norm

    def get_coeffs(self):
        return (self.pair1.lca.coeffs, self.pair2.lca.coeffs), \
               (self.pair1.lcb.coeffs, self.pair2.lcb.coeffs)

    def get_max_strain(self):
        return max(self.pair1.strain, self.pair2.strain)

    def todict(self):
        dct = {
            "natoms": self.get_natoms(),
            "coeffs_a": (self.pair1.lca.coeffs, self.pair2.lca.coeffs),
            "coeffs_b": (self.pair1.lcb.coeffs, self.pair2.lcb.coeffs),
            "internal_angle": self.get_internal_angle(),
            "twist_angle": self.get_twist_angle(),
            "max_strain": self.get_max_strain(),
            "norm_ratio": self.get_norm_ratio()
        }
        return dct


def get_atoms_and_stiffness(uid_a, uid_b, db):
    db = connect(db)
    info_a = db.get(uid=uid_a)
    info_b = db.get(uid=uid_b)
    atoms_a = info_a.toatoms()
    atoms_b = info_b.toatoms()
    stif_a = np.array([info_a.c_11, info_a.c_22, info_a.c_33])
    stif_b = np.array([info_b.c_11, info_b.c_22, info_b.c_33])
    return atoms_a, atoms_b, stif_a, stif_b


# Angle between two vectors, measured counter-clockwise.
def angle_between(v1, v2):
    # Moified sign function that returns 1 also if the value is 0
    def sign(num):
        if num >= 0:
            return 1
        return -1

    v1u = v1 / np.linalg.norm(v1)
    v2u = v2 / np.linalg.norm(v2)
    return sign(v2u[1]) * np.arccos(v2u[0]) - sign(v1u[1]) * np.arccos(v1u[0])


def get_approx_strain(norm1, norm2):
    return abs(norm1 - norm2) / min([norm1, norm2]) * 100


def area(v1, v2):
    return (v1[0] * v2[1] - v1[1] * v2[0])


def get_const(atoms_a, atoms_b):
    area_a = area(*atoms_a.cell[:2])
    area_b = area(*atoms_a.cell[:2])
    return len(atoms_a) / area_a + len(atoms_b) / area_b


# Generate all linear combinations for both lattices
def make_linear_combinations(atoms_a, atoms_b, max_coeff):
    rng = range(-max_coeff, max_coeff + 1)
    coeffs = [(m1, m2) for m1 in rng for m2 in rng if (m1, m2) != (0, 0)]
    lcs_a = [Lincomb(atoms_a, cfs) for cfs in coeffs]
    lcs_b = [Lincomb(atoms_b, cfs) for cfs in coeffs]
    return lcs_a, lcs_b


# Find linear combinations with matching moduli (low strain)
def match_components(lcs_a, lcs_b, max_strain):
    def sort(x): return x.norm
    lcs_a.sort(key=sort)
    lcs_b.sort(key=sort)
    matches = []
    firstmatch = 0
    for i in tqdm(range(len(lcs_a))):
        foundmatch = False
        for j in range(firstmatch, len(lcs_b)):
            strain = get_approx_strain(lcs_a[i].norm, lcs_b[j].norm)
            if strain <= max_strain:
                matches.append(Vecpair(lcs_a[i], lcs_b[j], strain))
                if not foundmatch:
                    foundmatch = True
                    firstmatch = j
    return matches


'''
# Look for supercells, i.e. LC pairs with matching rotation angle
def find_supercells(matches, tol_theta, const):
    matches.sort(key=lambda x: x.angle)
    supercells = []
    rng = len(matches)
    for i in tqdm(range(rng)):
        for j in range(i + 1, rng):
            if abs(matches[i].angle - matches[j].angle) >= tol_theta:
                break
            supercells.append(Supercell(matches[i], matches[j], const))
    return supercells
'''
# Look for supercells, i.e. LC pairs with matching rotation angle
def find_supercells(matches, tol_theta, const, twist_angle_interval):
    matches = [m for m in matches \
               if m.angle > twist_angle_interval[0] \
               and m.angle < twist_angle_interval[1]]
    matches.sort(key=lambda x: x.angle)
    supercells = []
    rng = len(matches)
    for i in tqdm(range(rng)):
        for j in range(i + 1, rng):
            if abs(matches[i].angle - matches[j].angle) >= tol_theta:
                break
            supercells.append(Supercell(matches[i], matches[j], const))
    return supercells


def generate_dataframe(cells):
    dict = {
        'twist_angle': [],
        'internal_angle': [],
        'natoms': [],
        'strain': [],
        'coeffs_a': [],
        'coeffs_b': [],
        'norm_ratio': []
    }

    for cell in cells:
        coeffs_a, coeffs_b = cell.get_coeffs()
        dict['twist_angle'].append(np.rad2deg(cell.get_twist_angle()))
        dict['internal_angle'].append(np.rad2deg(cell.get_internal_angle()))
        dict['natoms'].append(cell.get_natoms())
        dict['strain'].append(cell.get_max_strain())
        dict['coeffs_a'].append(coeffs_a)
        dict['coeffs_b'].append(coeffs_b)
        dict['norm_ratio'].append(cell.get_norm_ratio())
    return DataFrame.from_dict(dict)


def filter_dataframe(df, natoms, internal, asym):
    if asym:
        sym_filter = ''
    else:
        sym_filter = ' & norm_ratio >= 0.99 & norm_ratio <= 1.01'
    natoms_filter = f'natoms <= {natoms} & natoms > 0'
    internal_filter = f'internal_angle >= {internal[0]} & internal_angle <= {internal[1]}'
    df.query(natoms_filter + ' & ' + internal_filter + sym_filter, inplace=True)
    df.reset_index(drop=True, inplace=True)


@command('asr.findmoire')
@option('--uid-a', type=str)
@option('--uid-b', type=str)
@option('--max-coef', type=int,
        help='Max coefficient for linear combinations of the starting vectors')
@option('--tol-theta', type=float,
        help='Tolerance over rotation angle difference between matching vector pairs')
@option('--max-strain', type=float,
        help='Store only supercells with max (percent) strain lower than the specified one')
@option('--max-natoms', type=float,
        help='Store only supercells with lower number f atoms than the specified one')
@option('--twist-angle-interval', type=list,
        help='Lower and upper limits for the supercell twist angle (in degrees)')
@option('--min-internal-angle', type=float,
        help='Lower limit for the supercell internal angle (in degrees)')
@option('--max-internal-angle', type=float,
        help='Upper limit for the supercell internal angle (in degrees)')
@option('--keep-asymmetrical', type=bool, is_flag=True,
        help='Keep also cells with different cell vector lengths')
@option('--overwrite', type=bool, is_flag=True,
        help='Regenerate directory structure overwriting old files')
@option('--database', type=str,
        help='Path of the .db database file for retrieving structural information')
@option('--directory',
        help='Path of the .db database file for retrieving structural information')
def main(uid_a: str = None,
         uid_b: str = None,
         max_coef: int = 15,
         tol_theta: float = 0.05,
         max_strain: float = 1.0,
         max_natoms: int = 300,
         twist_angle_interval: list = [0, 360],
         min_internal_angle: float = 5.0,
         max_internal_angle: float = 175.0,
         keep_asymmetrical: bool = False,
         overwrite: bool = False,
         directory: Union[str, None] = None,
         #database: str = "/home/niflheim/steame/hetero-bilayer-project/databases/c2db.db"):
         database: str = "/home/niflheim2/cmr/databases/c2db/c2db-first-class-20240102.db"):

    layer_a, layer_b, stif_a, stif_b = get_atoms_and_stiffness(uid_a, uid_b, database)
    name_a = layer_a.get_chemical_formula()
    name_b = layer_b.get_chemical_formula()

    if not directory:
        directory = f"{name_a}-{name_b}"
    Path(directory).mkdir(exist_ok=True)
    if overwrite == True:
        try:
            Path(f"{directory}/moirecells.json").unlink()
            Path(f"{directory}/moirecells.cells").unlink()
        except:
            pass

    lcs_a, lcs_b = make_linear_combinations(layer_a, layer_b, max_coef)

    print('Matching components...')
    matches = match_components(lcs_a, lcs_b, max_strain)
    print(f"Obtained {len(matches)} LCs matches")

    print("\nSearching for supercells...")
    cells = find_supercells(matches, tol_theta, get_const(layer_a, layer_b), twist_angle_interval)
    print(f"\nFound a total of {len(cells)} supercells!\nFiltering results...")

    df = generate_dataframe(cells)
    filter_dataframe(df, max_natoms, [min_internal_angle,
                     max_internal_angle], keep_asymmetrical)
    print(f"\nFound {df.shape[0]} supercells that satisfy the requested conditions!!!")

    df.to_string(f"{directory}/cells.txt", header=True, index=True, float_format='%.2f')
    df.to_json(f"{directory}/cells.json", orient='index')
    dct = read_json(f"{directory}/cells.json")
    dct.update({
        'uid_a': uid_a,
        'uid_b': uid_b
    })
    write_json(f"{directory}/cells.json", dct)

    '''
    TODO:
    
    calcolare score per ogni cella:
    - basso strain
    - basso numero atomi
    - alta simmetria:
        - vettori cella hanno modulo simile
        - angolo vicino a 90, 60 o 120

    generare dataframe con supercelle e cagare fuori un qualche filename

    creare filtro avanzato per selezionare celle

    creare funzia per generare atoms objects a partire da selezione

    applicare funzia a top 10
    
    eliminare tutta la parte di creazione cartelle e sottocartelle
    '''

    '''
    cells_sorted = SortResults(cells, sort)
    SaveJson(cells, monos[i], monos[j], directory)
    save_human_readable(cells, uid_a, uid_b, directory)
    '''


if __name__ == '__main__':
    main().cli()
