from typing import List, Tuple
import numpy as np
from ase import Atoms


class AnyVector:
    pass


class ElementSet:
    # positions: List[Tuple[bool, np.array]]

    def __init__(self, movable_indices, positions):
        assert isinstance(positions, np.ndarray) or isinstance(positions, list)
        positions = np.array(positions, dtype=float)
        assert len(positions[0]) == 3, positions
        assert isinstance(movable_indices, list)
        assert all(isinstance(x, bool) for x in movable_indices)
        self._positions = list(zip(movable_indices, positions))

    def add_vec(self, vector):
        res = []
        for b, p in self._positions:
            if b:
                res.append(p + vector)
            else:
                res.append(p.copy())
        return res

    def get_positions(self):
        return [p.copy() for b, p in self._positions]

    def set_positions(self, value, v=None):
        if v is None:
            delta = np.array([0., 0., 0.])
        else:
            delta = v
        for t in value:
            assert type(t) == np.array or type(
                t) == np.ndarray, f"type:{type(t)}, t:{t}"

        self._positions = [(b, t + delta) for (b, _), t in zip(self._positions, value)]

    def get_data(self):
        return self._positions

    def copy(self):
        inds = [b for b, p in self._positions]
        pos = [p for b, p in self._positions]
        return ElementSet(inds.copy(), pos.copy())

    def remove_at(self, index):
        inds = [b for i, (b, p) in enumerate(self._positions) if i != index]
        pos = [p.copy() for i, (b, p) in enumerate(self._positions) if i != index]
        return ElementSet(inds.copy(), pos.copy())


class Material:
    def __init__(self, movable, atoms):
        assert len(movable) == len(atoms)
        els = {}
        sets = {}
        for b, atom in zip(movable, atoms):
            if atom.symbol not in els:
                els[atom.symbol] = []

            els[atom.symbol].append((b, atom.scaled_position))
        for k, v in els.items():
            bs = [b for b, _ in v]
            ps = [p for _, p in v]
            sets[k] = ElementSet(bs, ps)

        self.sets = sets


def mod(x, M):
    assert np.allclose(M, int(M))
    M = int(M)
    res = np.array([x[0] % M, x[1] % M, x[2] % M])

    # This is necessary because the mod operation
    # can fail to function properly for floats
    for i, v in enumerate(res):
        if np.allclose(abs(v), M):
            res[i] = 0.0
    return res


def equiv_w_vector(v: np.array, set1: ElementSet, set2: ElementSet) -> bool:
    added = set1.add_vec(v)
    remaining = set2.get_positions()
    for x in added:
        try:
            i = next(i for i, y in enumerate(remaining)
                     if np.allclose(mod(x - y, 1), 0.0))
        except StopIteration:
            return False
        remaining.pop(i)

    return True


def equiv_vector(set1: ElementSet, set2: ElementSet) -> Tuple[float, float]:
    """Return the translation vector which makes the two sets identical.

    The translation vector is folded back into the unit cell.
    i.e. if set1 = {(0.0, 0.0)}; set2 = {(0.5, 0.0)}
    return (0.5, 0.0)
    if set1 = {(0.0, 0.0)}; set2 = {(-0.5, 0.0)}
    return (0.5, 0.0)
    if set1 = {(0.0, 0.0)}; set2 = {(1.5, 0.0)}
    return (0.5, 0.0)

    return None if there is such vector. Can happen if we have 2 positions.
    e.g. if set1 = {(0.0, 0.0), (2/3, 0.0)}; set2 = {(0.0, 0.0), (1/2, 0.0)}
    return None
    """
    if len(set1.get_positions()) != len(set2.get_positions()):
        return None

    if not any(b for (b, p) in set1.get_data()):
        if any(b for (b, p) in set2.get_data()):
            return np.array([0, 0, 0])
        else:
            return AnyVector

    ref_ind, reference_pos = next((i, p)
                                  for i, (b, p) in enumerate(set1.get_data()) if b)
    if len(set1.get_positions()) == 1:
        deltavec = set2.get_data()[0][1] - reference_pos
        if np.allclose(deltavec[2], 0.0):
            return deltavec
        else:
            return None

    candidates = []
    for i2, (_, p2) in enumerate(set2.get_data()):
        v = mod(p2 - reference_pos, 1)

        _s1 = set1.removeAt(ref_ind)
        _s2 = set2.removeAt(i2)

        b = equiv_w_vector(v, _s1, _s2)
        if b:
            candidates.append(v)

    if len(candidates) > 0:
        candidates = [mod(c, 1) for c in candidates]
        index = np.argmin([np.linalg.norm(c) for c in candidates])
        if index > len(candidates) or index < 0:
            raise ValueError(f"{index}/{len(candidates)}")
        return candidates[index]

    return None


def slide_equivalent(slide_indices: List[int],
                     ats1: Atoms, ats2: Atoms) -> bool:
    """Calculate if two bilayers structures are sliding equivalent.

    Sliding is defined as translation in the xy-plane.

    Determine if we can go from mat2 to mat1 by sliding the atoms in mat1.
    """
    if ats1 == ats2:
        return np.array([0, 0, 0])
    if len(ats1) != len(ats2):
        return None

    mat1 = Material(slide_indices, ats1)
    mat2 = Material([False for _ in range(len(ats2))],
                    ats2)

    def mapply(i, v, set1, set2):
        if i == 0:
            v = equiv_vector(set1, set2)
        elif v is AnyVector:
            v = equiv_vector(set1, set2)
        elif v is not None:
            b = equiv_w_vector(v, set1, set2)
            if not b:
                v = None

        return i + 1, v

    vec = None
    i = 0
    for k, set1 in mat1.sets.items():
        set2 = mat2.sets[k]
        i, vec = mapply(i, vec, set1, set2)

    return vec


def invert(atoms, N):
    pos = atoms.positions

    mean_z = np.mean(pos[:, 2])

    relative_z = pos.copy()
    relative_z[:, 2] = relative_z[:, 2] - mean_z

    inverted_z = relative_z.copy()
    inverted_z[:, 2] *= -1

    inverted_z[:, 2] += mean_z
    assert (inverted_z[:, 2] >= 0).all()

    atoms2 = atoms.copy()
    atoms2.set_positions(inverted_z)

    dvec = atoms.positions[0] - atoms2.positions[N]
    atoms2.positions[:] += dvec
    atoms2.wrap()

    return atoms2


def test_slide_equiv(folder):
    """Analyse bilayer located in folder.

    Test whether the bilayer is equivalent to its z-inverted
    version through sliding.
    """
    from asr.utils.bilayerutils import construct_bilayer
    from ase.io import read
    bottom = read(f"{folder}/../structure.json")
    N_mono = len(bottom)

    bilayer = construct_bilayer(folder)
    inverted_bilayer = invert(bilayer, N_mono)

    bs = [i < N_mono for i in range(2 * N_mono)]
    b = slide_equivalent(bs, inverted_bilayer, bilayer)

    return b


def align_vector(bottom1, bottom2):
    from itertools import product
    deltavec = None
    for atom1, atom2 in product(bottom1, bottom2):
        if atom1.symbol == atom2.symbol:
            _b1 = bottom1.copy()
            _b2 = bottom2.copy()
            dvec = atom2.position - atom1.position
            if (_b1.translate(dvec)) == _b2:
                deltavec = dvec
                break
    return deltavec


def distance(a1, a2):
    from asr.database.rmsd import get_rmsd
    _v = get_rmsd(a1, a2)
    if _v is None:
        return 1000
    else:
        return _v


def get_slide_vector(bottom1, top1, bottom2, top2, t1_c, t2_c):
    from itertools import product
    from asr.stack_bilayer import atomseq

    if not atomseq(bottom1, bottom2):
        return None

    for atom1, atom2 in product(top1, top2):
        if atom1.symbol != atom2.symbol:
            continue
        dvec = atom2.position - atom1.position
        _top1 = top1.copy()
        _top1.translate(dvec)
        if atomseq(_top1, top2):
            return (bottom1.cell.scaled_positions(dvec)
                    + bottom2.cell.scaled_positions(np.array([t2_c[0], t2_c[1], 0]))
                    - bottom1.cell.scaled_positions(np.array([t1_c[0], t1_c[1], 0])))

    return None


def slide_vector_for_bilayers(folder1, folder2):
    from ase.io import read
    from asr.core import read_json

    bottom1 = read(f"{folder1}/../structure.json")
    top1 = read(f"{folder1}/toplayer.json")
    bottom2 = read(f"{folder2}/../structure.json")
    top2 = read(f"{folder2}/toplayer.json")
    data1 = read_json(f"{folder1}/translation.json")
    data2 = read_json(f"{folder2}/translation.json")
    t1_c = np.array(data1['translation_vector']).astype(float)
    t2_c = np.array(data2['translation_vector']).astype(float)

    return get_slide_vector(bottom1, top1, bottom2, top2, t1_c, t2_c)
