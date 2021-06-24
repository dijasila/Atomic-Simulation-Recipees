"""Fitted elemental reference energies."""
from asr.core import command, option, ASRResult
from ase.formula import Formula


class MaterialNotFoundError(Exception):
    pass


class DBAlreadyExistsError(Exception):
    pass


class ParseError(Exception):
    pass


def where(pred, ls):
    return list(filter(pred, ls))


def only(pred, ls):
    rs = where(pred, ls)
    assert len(rs) == 1
    return rs[0]


def single(pred, ls):
    rs = where(pred, ls)
    N = len(rs)
    assert N == 0 or N == 1
    if N == 1:
        return rs[0]
    else:
        return None


def select(pred, ls):
    return list(map(pred, ls))


def count(pred, ls):
    rs = select(pred, ls)
    return sum(rs)


def unique(ls, selector=None):
    if selector:
        rs = select(selector, ls)
    else:
        rs = ls

    return all(count(lambda x: x == y, rs) == 1 for y in rs)


def parse_reactions(reactionsstr):
    """Parse experimental HOF.

    Matches string in files looking like:

    Mo2S4 -0.03
    SO3 +0.1
    """
    import re

    with open(reactionsstr, "r") as f:
        data = f.read()

    lines = [line for line in data.split("\n") if line != ""]
    reactions = []

    splitter_re = r"(([A-Z]+[a-z]*[0-9]*)+)(\s)+([-+]?[0-9]+(\.[0-9]*)?)"
    for line in lines:
        tline = line.strip()
        match = re.match(splitter_re, tline)
        if match:
            form = match.group(1)
            energy = float(match.group(4))
            reactions.append((form, energy))
        else:
            raise ParseError(
                "Could not parse line" + ' "{}" in {}'.format(line, reactionsstr)
            )

    if not unique(reactions, lambda t: t[0]):
        bad = where(
            lambda y: count(lambda x: x[0] == y[1][0], reactions) > 1,
            enumerate(reactions),
        )
        raise ParseError("Same reaction was entered" + "multiple times: {}".format(bad))
    return reactions


def parse_refs(refsstr):
    """Parse elements to correct.

    Matches a file with:

    Ne
    O
    F
    """
    import re

    with open(refsstr, "r") as f:
        data = f.read()

    lines = [line for line in data.split("\n") if line != ""]
    refs = []

    parser_re = r"(^[A-Z]+[a-z]*[0-9]*$)"
    for line in lines:
        tline = line.strip()
        match = re.match(parser_re, tline)
        if match:
            form = match.group(1)
            refs.append(form)
        else:
            raise ParseError(
                "Could not parse line" + ' "{}" in {}'.format(line, refsstr)
            )

    if not unique(refs):
        bad = where(lambda y: count(lambda x: x == y[1], refs) > 1, enumerate(refs))
        raise ParseError(
            "Same reference" + "was entered multiple times: {}".format(bad)
        )
    return refs


def load_data(reactionsstr, refsstr):
    reacts = parse_reactions(reactionsstr)
    refs = parse_refs(refsstr)
    return reacts, refs


def elements_from_refs(refs):
    els = []
    for ref in refs:
        el = only(lambda t: True, Formula(ref).count().keys())
        els.append(el)
    return els


def multiply_formula(prod, j):
    form = Formula(prod)
    return Formula.from_dict({k: v * j for k, v in form.count().items()})


def safe_get(db, prod, query=''):
    result = None
    for j in range(50):
        formula = multiply_formula(prod, j + 1)
        try:
            q = ',' + query if query != '' else ''
            result = db.get("formula={}".format(formula) + q)
            break
        except Exception as e:
            if type(e) == KeyError:
                continue
            else:
                print("formula={}".format(formula) + q)
                raise e

    if result is None:
        raise MaterialNotFoundError("Could not find {} in db".format(prod))

    return result


def get_hof(db, formula, query='', row=None):
    elements = list(formula.count().keys())
    row = row or safe_get(db, str(formula), query=query)
    dbformula = Formula(str(row.formula))
    hof = row.energy
    for el in elements:
        elrow = safe_get(db, el)
        elformula = Formula(elrow.formula)
        number_el = only(lambda t: True, elformula.count().values())
        factor = dbformula.count()[el] / number_el
        hof -= factor * elrow.energy

    num_atoms = sum(dbformula.count().values())
    return hof / num_atoms


def get_dE_alpha(db, reactions, refs):
    from scipy import sparse

    alpha = sparse.lil_matrix((len(reactions), len(refs)))
    DE = sparse.lil_matrix((len(reactions), 1))

    for i1, (form, eexp) in enumerate(reactions):
        formula = Formula(form)
        hof = get_hof(db, formula)

        DE[i1, 0] = eexp - hof

        num_atoms = sum(formula.count().values())
        for i2, ref in enumerate(refs):
            reff = Formula(ref)
            el = only(lambda t: True, reff.count().keys())
            if el in formula.count().keys():
                alpha[i1, i2] = formula.count()[el] / num_atoms

    return DE, alpha


def minimize_error(dE, alpha):
    from scipy.sparse.linalg import spsolve
    import numpy as np

    b = -alpha.T.dot(dE)
    A = alpha.T.dot(alpha)

    dMu = spsolve(A, b)

    d = alpha.dot(dMu)
    error = dE.T.dot(dE) + 2 * dE.T.dot(alpha.dot(dMu)) + d.T.dot(d)
    error = np.sqrt(error / dE.shape[0])

    return dMu, error


def formulas_eq(form1, form2):
    if type(form1) == str:
        form1 = Formula(form1)
    if type(form2) == str:
        form2 = Formula(form2)
    return form1.stoichiometry()[:-1] == form2.stoichiometry()[:-1]


def create_corrected_db(newname, db, reactions, els_dMu):
    from ase.db import connect

    newdb = connect(newname)

    for row in db.select():
        formula = Formula(row.formula)
        el_dmu = single(lambda t: formulas_eq(t[0], formula), els_dMu)
        if el_dmu:
            el, dmu = el_dmu
            row.energy += formula.count()[el] * dmu
        newdb.write(row)


@command("asr.fere")
@option("--newdbname", help="Name of the new db file", type=str)
@option("--dbname", help="Name of the base db file", type=str)
@option(
    "--reactionsname",
    help="File containing reactions and energies with which to fit",
    type=str,
)
@option(
    "--referencesname",
    help="File containing the elements"
    + " whose references energies should be adjusted",
    type=str,
)
def main(
    newdbname: str = "newdb.db",
    dbname: str = "db.db",
    reactionsname: str = "reactions.txt",
    referencesname: str = "references.txt",
) -> ASRResult:
    from ase.db import connect
    import os
    import numpy as np

    if os.path.exists(newdbname):
        raise DBAlreadyExistsError
    reactions, refs = load_data(reactionsname, referencesname)

    db = connect(dbname)
    dE, alpha = get_dE_alpha(db, reactions, refs)

    dMu, error = minimize_error(dE, alpha)

    elements = elements_from_refs(refs)
    create_corrected_db(newdbname, db, reactions, list(zip(elements, dMu)))

    results = {
        "dbname": dbname,
        "newdbname": newdbname,
        "reactions": reactions,
        "refs": refs,
        "dE": np.array(dE.todense()),
        "alpha": str(alpha),
        "dMu": dMu,
        "error": error,
    }

    results["__key_descriptions__"] = {
        "dbname": "Name of base db",
        "newdbname": "Name of corrected db",
        "reactions": "Reactions and energies used to correct",
        "refs": "References that were adjusted",
        "dE": "Difference between target and initial HoFs",
        "alpha": "Alpha matrix",
        "dMu": "Adjustment of reference energies",
        "error": "RMSE after adjustment",
    }

    return results


if __name__ == "__main__":
    main.cli()
