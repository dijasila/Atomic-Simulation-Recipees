import typing
from asr.core import command, ASRResult, prepare_result, read_json
from pathlib import Path
from ase.io import read
from ase import Atoms
import numpy as np
import spglib
import os

@prepare_result
class Result(ASRResult):
    """Descriptor(s) of bilayer."""

    descriptor: str
    full_descriptor: str

    key_descriptions = dict(descriptor='A short descriptor of a stacking',
                            full_descriptor='A full descriptor of a stacking')


def convert_to_cartesian(matrix, cell):
    a1, a2, _ = cell
    a1 = a1[:2]
    a2 = a2[:2]
    b1 = np.array([a2[1], -a2[0]])
    b2 = np.array([a1[1], -a1[0]])
    b1 /= b1.dot(a1)
    b2 /= b2.dot(a2)
    assert np.allclose(b1.dot(a2), 0.0)
    assert np.allclose(b2.dot(a1), 0.0)

    N = np.array([a1, a2]).T
    B = np.array([b1, b2])
    
    return N.dot(matrix.dot(B))


def get_matrix_descriptor(atoms, matrix):
    cmatrix = convert_to_cartesian(matrix, atoms.cell)
    ## Decompose transformation matrix into form b^mc^n
    ## where b is a rotation matrix and c is a reflection
    
    ## Extract symmetries of cell
    cell = Atoms('C', positions=[[0, 0, 0]])
    cell.set_cell(atoms.get_cell())
    symmetries = spglib.get_symmetry(cell)
    ## Find a basis operation that is not the identity and not a reflection
    ## Find an inversion
    inversion_basis = next(convert_to_cartesian(y, atoms.cell)
                           for y in (x[:2, :2] for x in symmetries['rotations'])
                           if np.allclose(np.linalg.det(y), -1))
    
    isreflected = np.allclose(np.linalg.det(cmatrix), -1.0)
    
    bm = cmatrix.dot(np.linalg.inv(inversion_basis)) if isreflected else cmatrix

    
    # bm is a pure rotation.
    # It should have entries [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    # where theta = 2pi / n for some n.
    a1 = np.arctan2(-bm[0, 1], bm[0, 0])
    a2 = np.arctan2(bm[1, 0], bm[1, 1])

    assert np.allclose(abs(a1), abs(a2)), f"a1: {a1}, a2: {a2}\n\ncbm:{cbm}, isrefl:{isreflected}"
    
    if np.allclose(a1, 0.0):
        descriptor = 'I'
    else:
        n = 2 * np.pi / a1
        n = abs(round(n))
        descriptor = f'R<sup>z</sup><sub>{n}</sub>'
    if isreflected:
        descriptor += '_v' 

    return descriptor


def get_descriptor(folder=None, atoms=None):
    if folder is None:
        p = Path('.')
        folder = str(p.absolute())
    if atoms is None:
        p = Path(f'{folder}/structure.json').absolute()
        atoms = read(str(p))

    folder = [x for x in folder.split("/") if x != ""][-1]
    desc = "-".join(folder.split("-")[2:])
    
    # Extract matrix
    def tofloat(x):
        if x.startswith("M"):
            return - float(x[1:])
        else:
            return float(x)

    desc = desc.replace("--", "-M").replace("_-", "_M")
    parts = [p for p in desc.split("-") if p != ""]
    (a, b, c, d) = parts[0].split("_")
    matrix = np.array([[tofloat(a), tofloat(b)], [tofloat(c), tofloat(d)]])
    (tx, ty) = parts[-1].split("_")
    tx = tofloat(tx)
    ty = tofloat(ty)
    iz = "Iz" in desc

    descriptor = get_matrix_descriptor(atoms, matrix)
    
    if iz:
        descriptor += '_Iz'
    descriptor += f'_{tx:0.1f}_{ty:0.1f}'

    return descriptor


def set_first_class_info():
    """Determine if the material is first class and write to info.json"""
    from asr.setinfo import main as setinfo
    
    monolayerfolder = "../"
    p = Path(monolayerfolder).absolute()
    binding_data = []
    my_desc = get_descriptor()
    for sp in [x for x in p.iterdir() if x.is_dir()]:
        # Get binding data
        desc = get_descriptor(str(sp))
        binding_path = f"{sp}/results-asr.bilayer_binding.json"
        print(f"binding_path = {binding_path}")
        if os.path.exists(binding_path):
            data = read_json(binding_path)
            binding_data.append((desc, data["binding_energy"]))

    nmats = 5
    deltaE = 0.002
    cutoff = 0.15
    maxE = max(e for _, e in binding_data)
    selected = sorted(binding_data, key=lambda t: t[1])
    selected = [desc for desc, e in binding_data
                if abs(e - maxE) <= deltaE
                if e <= cutoff][:nmats]

    is_fst_class = my_desc in selected
    setinfo([('first_class_material', is_fst_class)])        
    return is_fst_class


@command(module='asr.bilayerdescriptor',
         returns=Result)
def main() -> Result:
    """Construct descriptors for the bilayer."""

    translation = read_json('translation.json')['translation_vector']
    transform = read_json('transformdata.json')

    rotation = transform['rotation']
    
    t_c = transform['translation'][:2] + translation

    p = "'" if not np.allclose(rotation, np.eye(3)) else ""
    B = 'B' if not np.allclose(t_c, 0.0) else 'A'

    descriptor = 'A' + B + p


    full_descriptor = get_descriptor()

    set_first_class_info()

    return Result.fromdata(descriptor=descriptor,
                           full_descriptor=full_descriptor)
    

    
