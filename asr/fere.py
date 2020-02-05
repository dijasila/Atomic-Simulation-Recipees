from asr.core import command, option


class MaterialNotFoundError(Exception):
    pass


class DBAlreadyExistsError(Exception):
    pass


def where(pred, ls):
    return list(filter(pred, ls))


def only(pred, ls):
    rs = where(pred, ls)
    assert len(rs) == 1
    return rs[0]


def load_data(reactionsstr, refsstr):
    import numpy as np
    reacts = [(x[0], float(x[1])) for x in np.load(reactionsstr)]
    refs = list(np.load(refsstr))
    return reacts, refs


def elements_from_refs(refs):
    from ase.formula import Formula
    els = []
    for ref in refs:
        el = only(lambda t: True, Formula(ref).count().keys())
        els.append(el)
    return els


def multiply_formula(prod, j):
    from ase.formula import Formula
    form = Formula(prod)
    return Formula.from_dict({k: v * j for k, v in form.count().items()})


def safe_get(db, prod):
    result = None
    for j in range(20):
        formula = multiply_formula(prod, j + 1)
        try:
            result = db.get("formula={}".format(formula))
            break
        except KeyError:
            continue
            
    if result is None:
        raise MaterialNotFoundError("Could not find {} in db".format(prod))

    return result


def get_dE_alpha(db, reactions, refs):
    from ase.formula import Formula
    from scipy import sparse
    
    alpha = sparse.lil_matrix((len(reactions), len(refs)))
    DE = sparse.lil_matrix((len(reactions), 1))
    
    for i1, (form, eexp) in enumerate(reactions):
        row = safe_get(db, form)
        e = row.de
        DE[i1, 0] = eexp - e
        
        formula = Formula(form)
        num_atoms = sum(formula.count().values())
        for i2, ref in enumerate(refs):
            reff = Formula(ref)
            el = only(lambda t: True, reff.count().keys())
            if el in formula.count().keys():
                alpha[i1, i2] = formula.count()[el] / num_atoms
                
    return DE, alpha


def minimize_error(dE, alpha):
    from scipy.sparse.linalg import spsolve
    
    b = alpha.T.dot(dE)
    A = alpha.T.dot(alpha)
    
    dMu = spsolve(A, b)
    
    d = alpha.dot(dMu)
    error = dE.T.dot(dE) - 2 * dE.T.dot(alpha.dot(dMu)) + d.T.dot(d)
    
    return dMu, error


def create_corrected_db(newname, db, reactions, els_dMu):
    from ase.formula import Formula
    from ase.db import connect

    newdb = connect(newname)
    
    for row in db.select():
        formula = Formula(row.formula)
        num_atoms = sum(formula.count().values())
        dde = 0
        for el, dMu in els_dMu:
            dde += formula.count().get(el, 0) * dMu / num_atoms
        row.de += dde
        
        newdb.write(row)


@command('asr.fere',
         resources='1:1h')
@option('--newdbname', help='Name of the new db file')
@option('--dbname', help='Name of the base db file')
@option('--reactionsname',
        help='File containing reactions and energies with which to fit')
@option('--referencesname',
        help='File containing the elements' +
        ' whose references energies should be adjusted')
def main(newdbname="newdb.db",
         dbname="db.db",
         reactionsname='reactions.npy',
         referencesname='references.npy'):
    from ase.db import connect
    import os
    if os.path.exists(newdbname):
        raise DBAlreadyExistsError
    reactions, refs = load_data(reactionsname, referencesname)

    db = connect(dbname)
    dE, alpha = get_dE_alpha(db, reactions, refs)

    dMu, error = minimize_error(dE, alpha)
    
    elements = elements_from_refs(refs)
    create_corrected_db(newdbname, db, reactions, list(zip(elements, dMu)))

    results = {'dbname': dbname,
               'newdbname': newdbname,
               'reactions': reactions,
               'refs': refs,
               'dE': str(dE),
               'alpha': str(alpha),
               'dMu': str(dMu),
               'error': str(error)}

    results['__key_descriptions__'] = \
        {'dbname': 'Name of base db',
         'newdbname': 'Name of corrected db',
         'reactions': 'Reactions and energies used to correct',
         'refs': 'References that were adjusted',
         'dE': 'Difference between target and initial HoFs',
         'alpha': 'Alpha matrix',
         'dMu': 'Adjustment of reference energies',
         'error': 'Error after adjustment'}
    
    return results


if __name__ == '__main__':
    main.cli()
