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
    # monolayer_uid: str
    translation: np.ndarray
    pointgroup_op: np.ndarray

    key_descriptions = dict(descriptor='A short descriptor of a stacking',
                            full_descriptor='A full descriptor of a stacking',
                            # monolayer_uid='UID of source monolayer',
                            translation=''.join(['Translation of top layer',
                                                 ' relative to bottom layer',
                                                 ' in scaled coordinates']),
                            pointgroup_op=''.join(['Point group operator',
                                                   ' of top layer relative',
                                                   ' to bottom layer.',
                                                   ' The matrix operators',
                                                   ' on the lattice vectors']))


def convert_to_cartesian(matrix, cell):
    a1, a2, _ = cell
    a1 = a1[:2]
    a2 = a2[:2]
    b1 = np.array([a2[1], -a2[0]])
    b2 = np.array([a1[1], -a1[0]])
    b1 /= b1 @ a1
    b2 /= b2 @ a2
    assert np.allclose(b1 @ a2, 0.0)
    assert np.allclose(b2 @ a1, 0.0)

    N = np.array([a1, a2]).T
    B = np.array([b1, b2])
    assert np.allclose(B @ N, np.eye(2))
    assert np.allclose(N @ B, np.eye(2)), N @ B

    return N @ matrix @ B


def get_matrix_descriptor(atoms, matrix):
    cmatrix = convert_to_cartesian(matrix, atoms.cell)
    # Decompose transformation matrix into form b^mc^n
    # where b is a rotation matrix and c is a reflection

    # Extract symmetries of cell
    cell = Atoms('C', positions=[[0, 0, 0]])
    cell.set_cell(atoms.get_cell())
    symmetries = spglib.get_symmetry(cell)
    # Find a basis operation that is not the identity and not a reflection
    # Find an inversion
    c2c = convert_to_cartesian
    isreflected = np.allclose(np.linalg.det(cmatrix), -1.0)
    if isreflected:
        ts = next((c2c(y, atoms.cell), y)
                  for y in (x[:2, :2] for x in symmetries['rotations'])
                  if np.allclose(np.linalg.det(c2c(y, atoms.cell)), -1))
        inversion_basis, invb = ts

    bm = cmatrix @ (np.linalg.inv(inversion_basis)) if isreflected else cmatrix

    # bm is a pure rotation.
    # It should have entries [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    # where theta = 2pi / n for some n.
    if np.allclose(bm, np.eye(2)):
        a1 = 0.0
        a2 = 0.0
    else:
        a1 = np.arctan2(-bm[0, 1], bm[0, 0])
        a2 = np.arctan2(bm[1, 0], bm[1, 1])

    msg = f"a1: {a1}, a2: {a2}\n\nbm:{bm}, isrefl:{isreflected}"
    assert np.allclose(abs(a1), abs(a2)), msg

    if np.allclose(a1, 0.0):
        descriptor = 'Id'
    else:
        n = 2 * np.pi / a1
        n = abs(round(n))
        descriptor = f'R<sup>z</sup><sub>{n}</sub>'
    if isreflected:
        descriptor += '_v'

    return descriptor


def get_descriptor_old(folder=None, atoms=None):
    if folder is None:
        p = Path('.')
        folder = str(p.absolute())
    if atoms is None:
        if not Path(f"{folder}/structure.json").is_file():
            p = Path(folder).resolve().parents[0]
            atoms = read(f"{p}/structure.json")
        else:
            p = Path(f'{folder}/structure.json').absolute()
            atoms = read(str(p))

    folder = [x for x in folder.split("/") if x != ""][-1]
    folder = folder.replace("--", "-M").replace("_-", "_M")
    desc = "-".join(folder.split("-")[2:])

    # Extract matrix
    def tofloat(x):
        if x.startswith("M"):
            return - float(x[1:])
        else:
            return float(x)

    parts = [p for p in desc.split("-") if p != ""]
    (a, b, c, d) = parts[0].split("_")
    matrix = np.array([[tofloat(a), tofloat(b)], [tofloat(c), tofloat(d)]])
    (tx, ty) = parts[-1].split("_")
    tx = tofloat(tx)
    ty = tofloat(ty)
    iz = "Iz" in desc

    descriptor = get_matrix_descriptor(atoms, matrix)

    if iz:
        descriptor = f'({descriptor}, Iz)'
    else:
        descriptor = f'({descriptor})'

    descriptor += f'_({tx:0.2f}, {ty:0.2f})'

    descriptor = descriptor.replace("_", "  ")
    return descriptor


def get_descriptor(folder=None, atoms=None):
    if folder is None:
        p = Path('.')
        folder = str(p.absolute())
    blfolder = folder

    if atoms is None:
        if not Path(f"{folder}/structure.json").is_file():
            p = Path(folder).resolve().parents[0]
            atoms = read(f"{p}/structure.json")
        else:
            p = Path(f'{folder}/structure.json').absolute()
            atoms = read(str(p))

    transformation_matrix_inplane = read_json(f"{blfolder}/transformdata.json")["rotation"]
    matrix = transformation_matrix_inplane[0:2,0:2]
    if np.allclose(transformation_matrix_inplane[2,2],-1): iz = True
    else: iz = False

    translation_inplane = read_json(f"{blfolder}/translation.json")["translation_vector"]
    transl = atoms.cell.scaled_positions(np.array([translation_inplane[0], translation_inplane[1], 0.0]))
    tx = transl[0]
    ty = transl[1]

    descriptor = get_matrix_descriptor(atoms, matrix)

    if iz:
        descriptor = f'({descriptor}, Iz)'
    else:
        descriptor = f'({descriptor})'

    descriptor += f'_({tx:0.2f}, {ty:0.2f})'

    descriptor = descriptor.replace("_", "  ")

    return descriptor


def set_first_class_info():
    """Determine if the material is first class and write to info.json."""
    from asr.setinfo import main as setinfo

    monolayerfolder = "../"
    p = Path(monolayerfolder).absolute()
    binding_data = []
    my_desc = get_descriptor()
    atoms = read(f"{p}/structure.json")
    for sp in [x for x in p.iterdir() if x.is_dir()]:
        # Get binding data
        desc = get_descriptor(str(sp), atoms=atoms)
        binding_path = f"{sp}/results-asr.bilayer_binding.json"
        if os.path.exists(binding_path):
            data = read_json(binding_path)
            energy = data["binding_energy"]
            if energy is not None:
                binding_data.append((desc, energy))

    if len(binding_data) == 0:
        setinfo([('first_class_material', False)])
        return False

    nmats = 50
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


def set_number_of_layers():
    from asr.setinfo import main as setinfo
    setinfo([('numberoflayers', 2)])


def set_monolayer_uid():
    from asr.setinfo import main as setinfo
    ml_uid = Path("..").resolve().name
    setinfo([('monolayer_uid', ml_uid)])


@command(module='asr.bilayerdescriptor',
         returns=Result)
def main() -> Result:
    """Construct descriptors for the bilayer."""
    translation = read_json('translation.json')['translation_vector']
    transform = read_json('transformdata.json')
    try: mirror = read_json('translation.json')['Bottom_layer_Mirror']
    except: mirror = False

    rotation = transform['rotation']

    t_c = transform['translation'][:2] + translation

    p = "'" if not np.allclose(rotation, np.eye(3)) else ""
    B = 'B' if not np.allclose(t_c, 0.0) else 'A'

    descriptor = 'A' + B + p

    full_descriptor = get_descriptor()
    if not mirror: full_descriptor = '(Id) - '+full_descriptor
    else: full_descriptor = '(Iz) - '+full_descriptor

    set_first_class_info()
    set_number_of_layers()
    set_monolayer_uid()

    return Result.fromdata(descriptor=descriptor,
                           full_descriptor=full_descriptor,
                           translation=t_c,
                           pointgroup_op=rotation)
