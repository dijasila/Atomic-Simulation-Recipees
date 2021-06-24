from asr.core import (
    command, option, argument, ASRResult, prepare_result, atomsopt
)
from ase.formula import Formula
import numpy as np
from typing import List, Tuple
from ase import Atoms


class CHCError(ValueError):
    pass


def gscheck(us):
    N = len(us)
    for i in range(N):
        for j in range(N):
            if i == j:
                assert not np.allclose(np.dot(us[i], us[j]), 0), np.dot(us[i], us[j])
            else:
                assert np.allclose(np.dot(us[i], us[j]), 0), np.dot(us[i], us[j])


def projuv(u, v):
    dp = np.dot(u, v)
    un2 = np.dot(u, u)
    return u * dp / un2


def mgs(u0):
    # Do Modified Gram-Schmidt to get N orthogonal vectors
    # starting from u
    assert u0.ndim == 1
    ndim = len(u0)
    es = np.eye(ndim)
    es = [es[:, j] for j in range(ndim)]
    lid = False
    for j in range(ndim):
        M = np.vstack([u0] + es[:j] + es[j + 1:]).T
        lid = not np.allclose(np.linalg.det(M), 0)
        if lid:
            break
    assert lid

    us = [u0]

    for j in range(1, ndim):
        u = M[:, j]
        for k in range(j):
            u -= projuv(us[k], u)

        us.append(u)

    gscheck(us)
    return us


def orthogonalize(us):
    nus = [us[0].copy()]

    for i in range(1, len(us)):
        u = us[i].copy()
        for j in range(i):
            u -= projuv(nus[j], u)
        nus.append(u)

    for i in range(len(nus)):
        for j in range(len(nus)):
            ip = np.dot(nus[i], nus[j])
            if i != j:
                if not np.allclose(ip, 0):
                    print("US:")
                    for u in us:
                        print(u)
                    print("NUS:")
                    for nu in nus:
                        print(nu)

                    raise ValueError("not fricking orthogonal")

    return nus


def mgsls(us):
    # Do Modified Gram-Schmidt to get a vector
    # that is orthogonal to us
    # us = [u.copy() for u in us]
    us = orthogonalize(us)
    ndim = len(us[0])
    nmissing = ndim - len(us)
    assert nmissing == 1, f"ndim = {ndim}, nvecs = {len(us)}"
    es = np.eye(ndim)
    es = [es[:, j] for j in range(ndim)]
    lid = False
    for j in range(ndim):
        M = np.vstack([us] + es[j:j + 1]).T
        lid = not np.allclose(np.linalg.det(M), 0)
        if lid:
            break
    assert lid

    newu = M[:, -1]
    for k in range(ndim - 1):
        newu -= projuv(us[k], newu)

    for k in range(ndim - 1):
        msg = f"newu: {newu}, ip: {np.dot(newu, us[k])}"
        assert np.allclose(np.dot(newu, us[k]), 0), msg

    us.append(newu)

    # gscheck(us)
    return newu


class Hyperplane:
    def __init__(self, pts, references):
        self.references = references
        self.ndim = len(pts[0])
        assert self.ndim == len(pts), f"ndim={self.ndim}, len(pts)={len(pts)}"
        self.pts = pts
        self.base_point = pts[0]
        self.vectors = []
        for j in range(1, len(pts)):
            vec = pts[j] - self.base_point
            assert not np.allclose(np.dot(vec, vec), 0)
            self.vectors.append(vec)

        if len(self.vectors) > 0:
            try:
                self.normal_vector = mgsls(self.vectors)
            except AssertionError as e:
                for x in pts:
                    print(x)

                print("-----")
                for y in self.vectors:
                    print(y)
                print("-----")
                print("-----")
                for ref in references:
                    print(ref.formula)

                raise e

    def contains(self, pt):
        if len(self.vectors) == 0:
            return True
        C = np.allclose(np.dot((pt - self.base_point), self.normal_vector), 0)
        return C

    def find_ts(self, P, contained=False):
        rP = P - self.base_point
        A = np.vstack(self.vectors).T
        assert np.allclose(A[:, 0], self.vectors[0])

        ts, errors, _, _ = np.linalg.lstsq(A, rP, rcond=None)
        if contained:
            assert np.allclose(errors, 0)

        return ts


class Line:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.base_point = pt1
        self.vector = pt2 - pt1
        self.ndim = len(pt1)
        self.normal_vectors = mgs(self.vector)[1:]
        assert len(self.normal_vectors) == self.ndim - 1

    def intersects(self, plane):
        # TODO Improve readability of this function
        assert self.ndim == len(plane.pts)
        if len(plane.vectors) == 0:
            return True
        normals = self.normal_vectors + [plane.normal_vector]
        A = np.vstack([N.T for N in normals])
        parallel = np.allclose(np.linalg.det(A), 0)
        if parallel:
            return plane.contains(self.base_point)

        bp = [np.dot(N, self.base_point) for N in self.normal_vectors]
        bp = bp + [np.dot(plane.normal_vector, plane.base_point)]
        b = np.array(bp)

        P = np.linalg.solve(A, b)

        s = self.find_s(P)
        ts = plane.find_ts(P, contained=True)
        if s < 0 or s > 1 or any((t < 0 or t > 1) for t in ts) or sum(ts) > 1:
            return False
        elif np.allclose(s, 0) or np.allclose(s, 1):
            # This check is not purely geometrical and thus
            # should be put somewhere else (in calculate_intermediates)
            # but I cant be bothered to this now.
            # Sorry, future person!
            return False
        else:
            return True

    def find_s(self, P):
        s = np.dot((P - self.base_point), self.vector)
        s = s / np.dot(self.vector, self.vector)
        return s


class Intermediate:
    def __init__(self, references, mat_reference, reactant_reference):
        self.references = references
        self.mat_ref = mat_reference
        self.reactant_ref = reactant_reference
        hform, x = self._get_hform_data()
        self.hform = hform
        self._x = x

    def to_dict(self):
        refdcts = [ref.to_dict() for ref in self.references]
        matdct = self.mat_ref.to_dict()
        reactdct = self.reactant_ref.to_dict()

        dct = {'refdcts': refdcts,
               'matdct': matdct,
               'reactdct': reactdct}
        return dct

    def from_dict(dct):
        if 'refdcts' not in dct:
            return LeanIntermediate.from_dict(dct)

        refdcts = dct['refdcts']
        matdct = dct['matdct']
        reactdct = dct['reactdct']
        refs = [Reference.from_dict(dct) for dct in refdcts]
        mat = Reference.from_dict(matdct)
        react = Reference.from_dict(reactdct)

        return Intermediate(refs, mat, react)

    @property
    def label(self):
        labels = map(lambda r: r.formula, self.references)
        x_lab = zip(self._x, labels)

        def s(x):
            return str(round(x, 2))

        label = ' + '.join([s(t[0]) + t[1] for t in x_lab])
        return label

    def to_result(self):
        thns = list(map(lambda r: (r.formula, r.hform), self.references))
        strs = [f'References: {thns}',
                f'Reactant content: {self.reactant_content}',
                f'Hform: {self.hform}']

        return strs

    def _get_hform_data(self):
        import numpy as np
        # Transform each reference into a vector
        # where entry i is count of element i
        # that is present in reference
        # Solve linear equation Ax = b
        # where A is matrix from reference vectors
        # and b is vector from mat_ref

        elements = self.mat_ref.to_elements()
        if len(elements) == 1:
            assert len(self.references) == 1, f'Els:{elements}, refs: {self.references}'
            hof = self.references[0].hform
            reac = self.reactant_ref.symbols[0]
            x = self.references[0].count[reac] / self.references[0].natoms

            return hof, [x]

        def ref2vec(_ref):
            _vec = np.zeros(len(elements))
            for i, el in enumerate(elements):
                _vec[i] = _ref.count[el]

            return _vec

        A = np.array([ref2vec(ref) for ref in self.references]).T

        b = ref2vec(self.mat_ref)

        if np.allclose(np.linalg.det(A), 0):
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            err = np.sum(np.abs(A.dot(x) - b))
            if err > 1e-4:
                for ref in self.references:
                    print(ref.formula, ref.hform)
                raise ValueError(f'Could not find solution.')
        else:
            x = np.linalg.solve(A, b)

        # hforms = np.array([ref.hform for ref in self.references])
        hforms = np.array([ref.energy for ref in self.references])

        counts = np.array([sum(ref.count.values()) for ref in self.references])
        norm = x.dot(counts)

        return np.dot(x, hforms) / norm, x

    @property
    def reactant_content(self):
        counters = zip(self._x, self.references)

        rform = self.reactant_ref.formula

        total_reactants = sum(map(lambda c: c[0] * c[1].count[rform],
                                  counters))
        total_matrefs = 1

        return total_reactants / (total_reactants + total_matrefs)


class LeanIntermediate:
    def __init__(self, mat_reference, reactant_reference,
                 reference):
        self.mat_ref = mat_reference
        self.reactant_ref = reactant_reference
        self.reference = reference
        self.hform = reference.hform
        react_symbol = reactant_reference.symbols[0]
        rc = reference.count[react_symbol] / reference.natoms
        assert not np.allclose(rc, 0.0)
        self.reactant_content = rc
        self.label = str(round(1 - rc, 2)) + reference.formula

    def to_result(self):
        thns = (self.reference.formula, self.reference.hform)
        strs = [f'Reference: {thns}',
                f'Reactant content: {self.reactant_content}',
                f'Hform: {self.hform}']

        return strs

    def to_dict(self):
        dct = {}
        dct["mat_ref"] = self.mat_ref.to_dict()
        dct["react_ref"] = self.reactant_ref.to_dict()
        dct["ref"] = self.reference.to_dict()

        return dct

    def from_dict(dct):
        mat_ref = Reference.from_dict(dct["mat_ref"])
        react_ref = Reference.from_dict(dct["react_ref"])
        ref = Reference.from_dict(dct["ref"])

        return LeanIntermediate(mat_ref, react_ref, ref)


class Reference:
    def __init__(self, formula, hform):
        from collections import defaultdict

        self.formula = formula
        self.hform = hform
        self.Formula = Formula(self.formula)
        self.energy = self.hform * self.natoms
        self.count = defaultdict(int)
        for k, v in self.Formula.count().items():
            self.count[k] = v
        self.symbols = list(self.Formula.count().keys())

    def __str__(self):
        """
        Make string version of object.

        Represent Reference by formula and heat of formation in a tuple.
        """
        return f'({self.formula}, {self.hform})'

    def __eq__(self, other):
        """
        Equate.

        Equate Reference-object with another
        If formulas and heat of formations
        are equal.
        """
        if type(other) != Reference:
            raise ValueError("Dont compare Reference to non-Reference")
            return False
        else:
            import numpy as np
            from asr.fere import formulas_eq
            feq = formulas_eq(self.formula, other.formula)
            heq = np.allclose(self.hform, other.hform)
            return feq and heq

    def __neq__(self, other):
        """
        Not Equal.

        Equate Reference-object with another
        if formulas and heat of formations
        are equal.
        """
        return not (self == other)

    def to_elements(self):
        return list(self.Formula.count().keys())

    def to_dict(self):
        dct = {'formula': self.formula,
               'hform': self.hform}
        return dct

    def from_dict(dct):
        formula = dct['formula']
        hform = dct['hform']
        return Reference(formula, hform)

    @property
    def natoms(self):
        return sum(self.Formula.count().values())


class ConvexHullReference(Reference):
    def __init__(self, *args, elements=None):
        super().__init__(*args)
        self.elements = elements
        self._construct_coordinates(elements)

    def _construct_coordinates(self, elements):
        coords = list(map(lambda e: self.count[e] / self.natoms, elements))
        self.coords = coords
        assert np.allclose(np.sum(self.coords), 1)

    def from_reference(ref, elements):
        return ConvexHullReference(ref.formula, ref.hform, elements=elements)

    def is_single(self):
        return any(np.allclose(c, 1) for c in self.coords)

    def __str__(self):
        """
        Get string version of ConvexHullRef.

        Represent Convex Hull Reference by
        1. Formula
        2. Heat of formation
        3. List of elements used in convex hull
        """
        msg = f'ConvexHullReference:' + f'\nFormula: {self.formula}'
        msg = msg + f'\nHform: {self.hform}' + f'\nElements: {self.elements}'
        return msg


def webpanel(result, context):
    from asr.database.browser import fig as asrfig

    fname = 'convexhullcut.png'

    panel = {'title': 'Convex Hull Cut',
             'columns': [[asrfig(fname)]],
             'plot_descriptions':
             [{'function': chcut_plot,
               'filenames': [fname]}]}

    return [panel]


def filrefs(refs):
    from asr.fere import formulas_eq
    nrefs = []
    visited = []
    for (form, v) in refs:
        seen = False
        for x in visited:
            if formulas_eq(form, x):
                seen = True
                break

        if seen:
            continue
        visited.append(form)

        vals = list(filter(lambda t: formulas_eq(t[0], form), refs))

        minref = min(vals, key=lambda t: t[1])

        nrefs.append(minref)

    return nrefs


def chcut_plot(context, fname):
    import matplotlib.pyplot as plt
    from ase import Atoms

    data = context.find_record('asr.chc').result
    mat_ref = Reference.from_dict(data['_matref'])

    if len(mat_ref.symbols) <= 2:
        refs = filrefs(data.get('_refs'))
        nrefs = []

        for (form, v) in refs:
            atoms = Atoms(form)
            e = v * len(atoms)
            nrefs.append((form, e))

        from ase.phasediagram import PhaseDiagram
        pd = PhaseDiagram(nrefs, verbose=False)
        plt.figure(figsize=(4, 3), dpi=150)
        pd.plot(ax=plt.gca(), dims=2, show=False)
        plt.savefig("./chcconvexhull.png")
        plt.close()

    mat_ref = Reference.from_dict(data['_matref'])
    reactant_ref = Reference.from_dict(data['_reactant_ref'])
    intermediates = [Intermediate.from_dict(im)
                     for im in data['_intermediates']]
    xs = list(map(lambda im: im.reactant_content, intermediates))
    es = list(map(lambda im: im.hform, intermediates))
    xs_es_ims = list(zip(xs, es, intermediates))
    xs_es_ims = sorted(xs_es_ims, key=lambda t: t[0])
    xs, es, ims = [list(x) for x in zip(*xs_es_ims)]
    labels = list(map(lambda im: im.label, ims))

    labels = [mat_ref.formula] + labels + [reactant_ref.formula]
    allxs = [0.0] + xs + [1.0]
    allxs = [round(x, 2) for x in allxs]
    labels = ['\n' + l if i % 2 == 1 else l for i, l in enumerate(labels)]
    labels = [f'{allxs[i]}\n' + l for i, l in enumerate(labels)]
    plt.plot([mat_ref.hform] + es + [0.0])
    plt.gca().set_xticks(range(len(labels)))
    plt.gca().set_xticklabels(labels)
    plt.xlabel(f'{reactant_ref.formula} content')
    plt.ylabel(f"Heat of formation")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')


@prepare_result
class Result(ASRResult):
    intermediates: List[Intermediate]
    material_info: str  # Reference?
    reactant: str
    mu_measure: float
    _matref: dict
    _intermediates: List[dict]
    _reactant_ref: dict
    _refs: List[Tuple[Formula, float]]

    key_descriptions = dict(
        intermediates='List of intermediates along convex hull cut.',
        material_info='Reference',
        reactant='Name of reactant to calculate cut against. E.g. "O".',
        mu_measure='Mu stability measure.',
        _matref='Material references.',
        _intermediates='List of intermediates.',
        _reactant_ref='Reference for reactant.',
        _refs='(formula, hform) list of relevant references.',
    )

    formats = {'webpanel2': webpanel}


@command('asr.chc')
@argument('dbs', nargs=-1, type=str)
@atomsopt
@option('-r', '--reactant', type=str,
        help='Reactant to add to convex hull')
def main(dbs: List[str],
         atoms: Atoms,
         reactant: str = 'O') -> Result:
    # Do type hints
    if len(dbs) == 0:
        raise ValueError('Must supply at least one database')

    from ase.db import connect
    from ase.formula import Formula
    dbs = [connect(db) for db in dbs]
    results = {}
    formula = str(atoms.symbols)
    elements = list(Formula(formula).count().keys())

    if reactant in elements:
        raise CHCError('Reactant is in elements')

    elements.append(reactant)

    mat_ref = results2ref(formula)

    reactant_ref = Reference(reactant, 0.0)
    references = [mat_ref, reactant_ref]
    append_references(elements, dbs, references)
    refs = convex_hull(references, mat_ref)
    if len(elements) > 2:
        intermediates = calculate_intermediates(mat_ref, reactant_ref, refs)
    else:
        intermediates = refs2ims(mat_ref, reactant_ref, refs)
    mum = mu_adjustment(mat_ref, reactant_ref, intermediates)

    results['intermediates'] = [im.to_result() for im in intermediates]
    results['material_info'] = str(mat_ref)
    results['reactant'] = reactant
    results['mu_measure'] = mum
    results['_matref'] = mat_ref.to_dict()
    results['_intermediates'] = [im.to_dict() for im in intermediates]
    results['_reactant_ref'] = reactant_ref.to_dict()
    results['_refs'] = [(ref.formula, ref.hform) for ref in references]

    return results


def results2ref(formula):
    from asr.core import read_json
    data = read_json("results-asr.convex_hull.json")
    return Reference(formula, data["hform"])


def get_hof(formula, energy, db):
    from ase.formula import Formula
    formula = Formula(formula)
    elements = list(formula.count().keys())
    hof = energy
    for el in elements:
        for row in db.select(f"{el}, ns=1"):
            _formula = Formula(row.formula)
            nels = len(list(_formula.count().keys()))
            if nels > 1:
                continue
            energy_per_el = row.energy / sum(_formula.count().values())
            hof += - formula.count()[el] * energy_per_el
            break

    return hof / sum(formula.count().values())


def row2ref(row, dbs):
    if hasattr(row, "hform"):
        return Reference(row.formula, row.hform)
    elif hasattr(row, "de"):
        return Reference(row.formula, row.de)
    elif hasattr(row, "hof"):
        return Reference(row.formula, row.hof)
    else:
        # from asr.fere import get_hof
        # from ase.formula import Formula
        # hof = get_hof(dbs[0], Formula(row.formula), row=row)
        hof = get_hof(row.formula, row.energy, dbs[0])
        return Reference(row.formula, hof)


def convex_hull(references, mat_ref):
    # Remove materials not on convex hull, except for formula
    from ase.phasediagram import PhaseDiagram
    pd = PhaseDiagram([(ref.formula, ref.energy) for ref in references],
                      verbose=False)
    hull = pd.hull
    filtered_refs = [references[i] for i, x in enumerate(hull) if x]
    # filtered_refs = []
    # for i, x in enumerate(hull):
    #     if x:
    #         filtered_refs.append(references[i])

    if not any(x == mat_ref for x in filtered_refs):
        filtered_refs.append(mat_ref)

    return filtered_refs


def append_references(elements, dbs, references):

    def _refin(ref, refls):
        for other_ref in refls:
            if other_ref == ref:
                return True
        return False

    def _elementcheck(row):
        from ase.formula import Formula
        formula = Formula(row.formula)
        _elements = list(formula.count().keys())
        return all(el in elements for el in _elements)

    selected_refs = []
    for element in elements:
        for db in dbs:
            for row in db.select(element):
                if not _elementcheck(row):
                    continue
                ref = row2ref(row, dbs)
                if _refin(ref, selected_refs):
                    continue
                selected_refs.append(ref)

    references.extend(selected_refs)

    return


def mu_adjustment(mat_ref, reactant_ref, intermediates):
    def f(im):
        x = (im.hform - mat_ref.hform)
        if np.allclose(im.reactant_content, 0):
            print("Bling blong")
            print(im.reactant_ref.formula)
            for i, ref in enumerate(im.references):
                print(f"Ref {i}:", ref.formula)
            raise ValueError('An Intermediate has 0 reactant content')
        x /= im.reactant_content
        return x

    adjustments = list(map(f, intermediates))
    return min(adjustments, default=0.0)


def get_coords(ref, elements):
    # Calculate relative content of each element in elements
    coords = list(map(lambda e: ref.count[e] / ref.natoms, elements))

    return (ref.formula, coords, ref.hform)


def calculate_intermediates(mat_ref, reactant_ref, refs):
    import numpy as np
    # Take out refs that consists of a single element and have
    # positive heat of formation. They will never be on the hull
    # but may destabilize hull algorithm.
    _refs = [r for r in refs if not (r.hform > 0 and len(r.symbols) == 1)]
    _refs = [mat_ref] + _refs + [reactant_ref]

    # Ordered list of unique elements
    elements = list(mat_ref.count.keys()) + [reactant_ref.formula]
    # elements = list(set(flatten(map(lambda r: r.to_elements(), _refs))))

    chrefs = [ConvexHullReference.from_reference(ref, elements)
              for ref in _refs]

    if any(r.is_single() and r.hform < 0 for r in chrefs):
        raise ValueError('Cannot have reference phase with negative HoF')

    # Line and planes are representation of the geometrical objects
    # plus information needed for this specific algorithm
    # e.g. heat of formation and chemical formula
    line, planes = convex_hull_planes(chrefs, mat_ref.formula, reactant_ref.formula)
    ims = []
    for plane in planes:
        if line.intersects(plane):
            refs = plane.references
            if mat_ref in refs:
                continue
            im = Intermediate(refs, mat_ref, reactant_ref)
            ims.append(im)

    ims = [im for im in ims if not np.allclose(im.reactant_content, 0)]

    return ims


def convex_hull_planes(chrefs, mat_formula, react_formula):
    import numpy as np
    from scipy.spatial import ConvexHull
    if chrefs[0].formula != mat_formula:
        msg = f'Material must be first in convex hull refs:'
        msg = msg + f' {(chrefs[0].formula, mat_formula)}'
        raise ValueError(msg)

    hull_coords = list(map(lambda r: r.coords[1:] + [r.hform], chrefs))

    hull = ConvexHull(hull_coords)

    # Equations contains a list of normal vectors and
    # offsets for the facet planes.
    # i.e. eqs[i] is [normalvector, offset] for facet i
    # We assume
    # (like ase.phasediagram.PhaseDiagram) that the normal vectors
    # are outward-pointing but this is not always true,
    # see http://www.qhull.org/html/qh-faq.htm#orient
    eqs = hull.equations

    # Get facet that points "downwards" in energy directions
    # This depends on energy being the last dimension
    _onhull = eqs[:, -2] < 0
    # hull.simplices is a list of tuples
    # The tuples are the indices of the cornes of the
    # simplices.
    simplex_indices = hull.simplices[_onhull]
    onhull = np.zeros(len(hull.points), bool)
    for simplex in simplex_indices:
        onhull[simplex] = True

    points = hull.points
    line = Line(points[0, :-1], points[-1, :-1])

    planes = []
    plane_inds = []
    for indices in simplex_indices:
        for i in range(len(indices)):
            # Remove point i from simplex.
            # This leaves us with a hyperplane
            # of dimension 1 lower than embedding space.
            ind = list(indices[:i]) + list(indices[i + 1:])
            # Check whether this tuple has already been considered
            # up to permutations
            if _permutecontain(ind, plane_inds):
                continue
            plane_inds.append(ind)

    for ind in plane_inds:
        refs = [chrefs[j] for j in ind]
        # Exclude if points are collinear
        if is_collinear(points[ind, :-1], ind, points, eqs, simplex_indices, onhull):
            continue
        plane = Hyperplane(points[ind, :-1], refs)
        planes.append(plane)

    return line, planes


def is_collinear(pts, ind, apts, eqs, sis, oh):
    # Return whether pts form a
    # N - 1 dimensional hyperplane
    # N = len(pts)
    npts = [pt - pts[0] for pt in pts[1:]]
    ortho_pts = orthogonalize(npts)
    if any(np.allclose(np.linalg.norm(pt), 0) for pt in ortho_pts):
        return True
    else:
        return False


def is_independent(v1, v2):
    f = None
    for x1, x2 in zip(v1, v2):
        if np.allclose(x2, 0) and np.allclose(x1, 0):
            continue
        elif np.allclose(x2, 0):
            return True
        elif np.allclose(x1, 0):
            return True
        else:
            if f is None:
                f = x1 / x2
            else:
                if not np.allclose(f, x1 / x2):
                    return True

    return False


def refs2ims(mat_ref, reactant_ref, refs):
    from asr.fere import formulas_eq
    ims = []
    for ref in refs:
        if formulas_eq(ref.formula, mat_ref.formula):
            continue
        if formulas_eq(ref.formula, reactant_ref.formula):
            continue

        lim = LeanIntermediate(mat_ref, reactant_ref, ref)

        ims.append(lim)

    return ims


def _permutecontain(t, tls):
    return any(tuplespermuted(item, t) for item in tls)


def tuplespermuted(t1, t2):
    def count(x, ite):
        return sum(map(lambda t: t == x, ite))

    for item in t1:
        if item not in t2:
            return False
        elif count(item, t1) != count(item, t2):
            return False
    return True


if __name__ == "__main__":
    main.cli()
