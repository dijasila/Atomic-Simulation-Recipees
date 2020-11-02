from typing import List, Tuple
import numpy as np
from ase import Atoms


class ElementSet:
    # positions: List[Tuple[bool, np.array]]

    def __init__(self, movable_indices, positions):
        assert type(positions) == np.array or type(positions) == list
        positions = np.array(positions, dtype=float)
        assert len(positions[0]) == 3, positions
        assert type(movable_indices) == list
        assert all(type(x) == bool for x in movable_indices)
        self._positions = list(zip(movable_indices, positions))

    def add_vec(self, vector):
        res = []
        for b, p in self._positions:
            if b:
                res.append(p.copy() + vector)
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

    def removeAt(self, index):
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
        return True
    if len(ats1) != len(ats2):
        return False

    mat1 = Material(slide_indices, ats1)
    mat2 = Material([False for _ in range(len(ats2))],
                    ats2)

    for k, set1 in mat1.sets.items():
        set2 = mat2.sets[k]
        vec = equiv_vector(set1, set2)

        if vec is not None:
            return True

    return False
