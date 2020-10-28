from dataclasses import dataclass, astuple
from typing import List, Tuple, Dict
import numpy as np
from ase import Atoms

@dataclass
class ElementSet:
    positions: List[np.array]
    
    def add_vec(self, vector):
        return [p.copy() + vector for p in self.positions]


@dataclass
class Material:
    elements: Dict[str, ElementSet]

def mod(x, M):
    assert np.allclose(M, int(M))
    M = int(M)
    res = np.array([x[0] % M, x[1] % M])

    # This is necessary because the mod operation
    # can fail to function properly for floats
    for i, v in enumerate(res):
        if np.allclose(abs(v), M):
            res[i] = 0.0
    return res
        

def equiv_w_vector(v: np.array, set1: ElementSet, set2: ElementSet) -> bool:
    added = [p.copy() + v for p in set1.positions]
    remaining = set2.positions.copy()
    for x in added:
        try:
            i = next(i for i, y in enumerate(remaining) if np.allclose(mod(x - y, 1), 0.0))
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
    if len(set1.positions) != len(set2.positions):
        return None

    candidates = []
    for i1, p1 in enumerate(set1.positions):
        v = mod(set2.positions[0] - p1, 1)
        
        _s1 = ElementSet(set1.positions[:i1] + set1.positions[i1+1:])
        _s2 = ElementSet(set2.positions[1:])

        b = equiv_w_vector(v, _s1, _s2)
        if b:
            candidates.append(v)
    
    if len(candidates) > 0:
        candidates = [mod(c, 1) for c in candidates]
        return candidates[np.argmin([np.linalg.norm(c) for c in candidates])]
    
    return None


def slide_equivalent(mat1: Atoms, mat2: Atoms) -> bool:
    raise NotImplementedError
