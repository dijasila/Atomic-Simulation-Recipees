from asr.core import command, option
import numpy as np


def gscheck(us):
    N = len(us)
    for i in range(N):
        for j in range(N):
            if i == j:
                assert not np.allclose(np.dot(us[i], us[j]), 0)
            else:
                assert np.allclose(np.dot(us[i], us[j]), 0)


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


def mgsls(us):
    # Do Modified Gram-Schmidt to get a vector
    # that is orthogonal to us
    us = [u.copy() for u in us]
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

    us.append(newu)

    gscheck(us)
    return us


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

        self.normal_vector = mgsls(self.vectors)[-1]

    def contains(self, pt):
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
        assert self.ndim == len(plane.pts)
        normals = self.normal_vectors + [plane.normal_vector]
        NM = np.vstack(normals).T
        parallel = np.allclose(np.linalg.det(NM), 0)
        if parallel:
            return plane.contains(self.base_point)

        A = np.vstack([N.T for N in normals])

        bp = [np.dot(N, self.base_point) for N in self.normal_vectors]
        bp = bp + [np.dot(plane.normal_vector, plane.base_point)]
        b = np.array(bp)

        P = np.linalg.solve(A, b)

        s = self.find_s(P)
        ts = plane.find_ts(P, contained=True)
        if s < 0 or s > 1 or any((t < 0 or t > 1) for t in ts):
            return False
        elif np.allclose(s, 0) or np.allclose(s, 1):
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

        def ref2vec(_ref):
            elements = self.mat_ref.to_elements()
            _vec = np.zeros(len(elements))
            for i, el in enumerate(elements):
                _vec[i] = _ref.count[el]

            return _vec

        A = np.array([ref2vec(ref) for ref in self.references]).T

        assert not np.allclose(np.linalg.det(A), 0)
        b = ref2vec(self.mat_ref)

        x = np.linalg.solve(A, b)

        hforms = np.array([ref.hform for ref in self.references])

        return np.dot(x, hforms), x

    @property
    def reactant_content(self):
        counters = zip(self._x, self.references)

        rform = self.reactant_ref.formula

        total_reactants = sum(map(lambda c: c[0] * c[1].count[rform],
                                  counters))
        total_matrefs = 1

        return total_reactants / (total_reactants + total_matrefs)


class Reference:
    def __init__(self, formula, hform):
        from ase.formula import Formula
        from collections import defaultdict

        self.formula = formula
        self.hform = hform
        self.Formula = Formula(self.formula)
        self.count = defaultdict(int)
        for k, v in self.Formula.count().items():
            self.count[k] = v

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
            return False
        else:
            import numpy as np
            feq = self.formula == other.formula
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

    def from_reference(ref, elements):
        return ConvexHullReference(ref.formula, ref.hform, elements=elements)

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


def webpanel(row, key_descriptions):
    from asr.database.browser import fig as asrfig

    fname = './convexhullcut.png'

    panel = {'title': 'Convex Hull Cut',
             'columns': [asrfig(fname)],
             'plot_descriptions':
             [{'functions': chcut_plot,
               'filenames': [fname]}]}

    return [panel]


def chcut_plot(row, *args):
    data = row.data.get('results-asr.chc.json')
    mat_ref = Reference.from_dict(data['_matref'])
    reactant_ref = Reference.from_dict(data['_reactant_ref'])
    intermediates = [Intermediate.from_dict(im)
                     for im in data['_intermediates']]

    import matplotlib.pyplot as plt
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
    plt.savefig('./convexhullcut.png', bbox_inches='tight')


@command('asr.chc',
         requires=['structure.json',
                   'results-asr.convex_hull.json'],
         webpanel=webpanel)
@option('--db', type=str,
        help='ASE DB containing the references')
@option('-r', '--reactant', type=str,
        help='Reactant to add to convex hull')
def main(db='references.db', reactant='O'):
    from ase.db import connect
    db = connect(db)
    results = {}
    formula, elements = read_structure('structure.json')
    assert reactant not in elements
    elements.append(reactant)

    mat_ref = results2ref(formula)
    reactant_ref = Reference(reactant, 0.0)
    references = [mat_ref]

    append_references(elements, db, references)

    refs = convex_hull(references, mat_ref)

    intermediates = calculate_intermediates(mat_ref, reactant_ref, refs)
    results['intermediates'] = [im.to_result() for im in intermediates]

    mum = mu_adjustment(mat_ref, reactant_ref, intermediates)

    results['material_info'] = str(mat_ref)
    results['reactant'] = reactant
    results['mu_measure'] = mum
    results['_matref'] = mat_ref.to_dict()
    results['_intermediates'] = [im.to_dict() for im in intermediates]
    results['_reactant_ref'] = reactant_ref.to_dict()

    return results


def read_structure(fname):
    from ase.io import read
    from ase.formula import Formula
    atoms = read(fname)
    formula = str(atoms.symbols)
    elements = list(Formula(formula).count().keys())

    return formula, elements


def results2ref(formula):
    from asr.core import read_json
    data = read_json("results-asr.convex_hull.json")
    return Reference(formula, data["hform"])


def row2ref(row):
    if hasattr(row, "hform"):
        return Reference(row.formula, row.hform)
    elif hasattr(row, "de"):
        return Reference(row.formula, row.de)
    else:
        raise ValueError("No recognized Heat of Formation key")


def convex_hull(references, mat_ref):
    # Remove materials not on convex hull, except for formula
    from ase.phasediagram import PhaseDiagram
    pd = PhaseDiagram([(ref.formula, ref.hform) for ref in references],
                      verbose=False)
    filtered_refs = []
    hull = pd.hull
    for i, x in enumerate(hull):
        if x:
            filtered_refs.append(references[i])

    if not any(x == mat_ref for x in filtered_refs):
        filtered_refs.append(mat_ref)

    return filtered_refs


def append_references(elements, db, references):
    # For each material in DB add material to references
    # if all elements in material are in elements-list
    # and material is not already in db??

    def rowin(_row, _rowls):
        _ref = row2ref(_row)
        for _orow in _rowls:
            _oref = row2ref(_orow)
            if _oref == _ref:
                return True

        return False

    def elementcheck(_row):
        from ase.formula import Formula
        _formula = Formula(_row.formula)
        _elements = list(_formula.count().keys())
        return all(_el in elements for _el in _elements)

    selected_rows = []
    for element in elements:
        for row in db.select(element):
            if rowin(row, selected_rows):
                continue
            if not elementcheck(row):
                continue
            selected_rows.append(row)

    new_refs = map(lambda r: row2ref(r), selected_rows)

    references.extend(new_refs)

    return


def mu_adjustment(mat_ref, reactant_ref, intermediates):
    def f(im):
        x = (mat_ref.hform - im.hform)
        x /= im.reactant_content
        return x

    adjustments = map(lambda im: f(im), intermediates)
    return max(adjustments)


def get_coords(ref, elements):
    # Calculate relative content of each element in elements
    coords = list(map(lambda e: ref.count[e] / ref.natoms, elements))

    return (ref.formula, coords, ref.hform)


def calculate_intermediates(mat_ref, reactant_ref, refs):
    reactant = reactant_ref.formula
    _refs = [mat_ref] + refs + [Reference(reactant, 0.0)]

    # Ordered list of unique elements
    elements = list(mat_ref.count.keys()) + [reactant]
    # elements = list(set(flatten(map(lambda r: r.to_elements(), _refs))))

    chrefs = [ConvexHullReference.from_reference(ref, elements)
              for ref in _refs]

    # Dont like this because ref object is used differently different places
    # And data content varies with time
    # for ref in refs:
    #     ref.construct_coordinates(elements)

    # Line and planes are representation of the geometrical objects
    # plus information needed for this specific algorithm
    # e.g. heat of formation and chemical formula
    line, planes = convex_hull_planes(chrefs, mat_ref.formula, reactant)
    ims = []
    for plane in planes:
        if line.intersects(plane):
            refs = plane.references
            im = Intermediate(refs, mat_ref, Reference(reactant, 0.0))
            ims.append(im)

    return ims


def convex_hull_planes(chrefs, mat_formula, react_formula):
    import numpy as np
    from scipy.spatial import ConvexHull

    hull_coords = list(map(lambda r: r.coords[1:] + [r.hform], chrefs))

    hull = ConvexHull(hull_coords)

    # Equations contains a list of normal vectors and
    # offsets for the facet planes. We assume
    # (like ase.phasediagram.PhaseDiagram that the normal vectors
    # are outward-pointing but this is not always true,
    # see http://www.qhull.org/html/qh-faq.htm#orient
    eqs = hull.equations

    # Get facet that points "downwards" in energy directions
    # This depends on energy being the last dimension
    _onhull = eqs[:, -2] < 0
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
            ind = list(indices[:i]) + list(indices[i + 1:])
            if _permutecontain(ind, plane_inds):
                continue
            plane_inds.append(ind)

    for ind in plane_inds:
        refs = [chrefs[j] for j in ind]
        plane = Hyperplane(points[ind, :-1], refs)
        planes.append(plane)

    return line, planes


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
