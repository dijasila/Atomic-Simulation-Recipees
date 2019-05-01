from asr.utils import command
from asr.utils.prototype import get_symmetry_id


@command('asr.quickinfo')
def main():
    """Get quick information about structure based on start.traj"""
    from random import randint
    from ase.io import read, jsonio
    from pathlib import Path
    from asr.utils import has_inversion, get_reduced_formula
    import json

    fnames = list(Path('.').glob('start.*'))
    assert len(fnames) == 1, fnames
    atoms = read(str(fnames[0]))
    info = {}

    folder = Path().cwd()
    info['folder'] = str(folder)

    # Determine magnetic state
    def get_magstate(a):
        magmom = a.get_magnetic_moment()
        if abs(magmom) > 0.02:
            return 'fm'

        magmoms = a.get_magnetic_moments()
        if abs(magmom) < 0.02 and abs(magmoms).max() > 0.1:
            return 'afm'

        # Material is essentially non-magnetic
        return 'nm'

    try:
        magstate = get_magstate(atoms)
    except RuntimeError:
        magstate = 'nm'
    info['magstate'] = magstate

    # Are forces/stresses known?
    try:
        f = atoms.get_forces()
        s = atoms.get_stress()[:2]
        fmax = ((f**2).sum(1).max())**0.5
        smax = abs(s).max()
        info['fmax'] = fmax
        info['smax'] = smax
    except RuntimeError:
        pass

    formula = atoms.get_chemical_formula(mode='metal')
    stoichimetry = get_reduced_formula(formula, stoichiometry=True)
    info['formula'] = formula
    info['stoichiometry'] = stoichimetry
    info['has_inversion_symmetry'] = has_inversion(atoms)

    def coarsesymmetries(a):
        from gpaw.symmetry import Symmetry
        cell_cv = a.get_cell()
        tol = 0.01  # Tolerance for coarse symmetries
        coarsesymmetry = Symmetry(
            a.get_atomic_numbers(),
            cell_cv,
            tolerance=tol,
            symmorphic=False,
            rotate_aperiodic_directions=True,
            translate_aperiodic_directions=True,
            time_reversal=True)
        coarsesymmetry.analyze(a.get_scaled_positions())
        return (coarsesymmetry.op_scc, coarsesymmetry.ft_sc)

    op_scc, ft_sc = coarsesymmetries(atoms)
    symmetry = [(op_cc.tolist(), ft_c.tolist())
                for op_cc, ft_c in zip(op_scc, ft_sc)]
    info['symmetries'] = symmetry

    try:
        import spglib
    except ImportError:
        pass
    else:
        sg, number = spglib.get_spacegroup(atoms, symprec=1e-4).split()
        number = int(number[1:-1])
        info['spacegroup'] = sg

    # Find prototype
    try:
        info['prototype'] = get_symmetry_id(atoms, symprec=0.5)
    except (OSError, ModuleNotFoundError):
        pass

    # Set temporary uid.
    # Will be changed later once we know the prototype.
    uid = '{}-X-{}-{}'.format(formula, magstate, randint(2, 9999999))
    info['uid'] = uid

    json.dump(info, open('quickinfo.json', 'w'), cls=jsonio.MyEncoder)

    return info


def collect_data(atoms):
    """Collect quick info to database"""
    from pathlib import Path
    p = Path('quickinfo.json')
    if not p.is_file():
        return {}, {}, {}

    import numpy as np
    import json

    data = {}
    kvp = {}
    key_descriptions = {}

    info = json.loads(p.read_text())

    exclude = ['symmetries', 'formula']
    for key in info:
        if key in exclude:
            continue
        kvp[key] = info[key]

    # Key-value-pairs:
    data['info'] = info
    if 'magstate' in kvp:
        kvp['magstate'] = kvp['magstate'].upper()
        kvp['is_magnetic'] = kvp['magstate'] != 'NM'

    if (atoms.pbc == [True, True, False]).all():
        kvp['cell_area'] = abs(np.linalg.det(atoms.cell[:2, :2]))

    key_descriptions = {
        'magstate': ('Magnetic state', 'Magnetic state', ''),
        'is_magnetic': ('Magnetic', 'Material is magnetic', ''),
        'cell_area': ('Area of unit-cell', '', 'Ang^2'),
        'has_invsymm': ('Inversion symmetry', '', ''),
        'uid': ('Identifier', '', ''),
        'stoichiometry': ('Stoichiometry', '', ''),
        'spacegroup': ('Space group', 'Space group', ''),
        'prototype': ('Prototype', '', '')}

    return kvp, key_descriptions, data


def webpanel(row, key_descriptions):
    from ase.db.summary import ATOMS, UNITCELL
    from asr.utils.custom import table

    stabilities = {1: 'low', 2: 'medium', 3: 'high'}
    basictable = table(row, 'Property', [
        'prototype', 'class', 'spacegroup', 'gap', 'magstate', 'ICSD_id',
        'COD_id'
    ], key_descriptions, 2)
    rows = basictable['rows']
    codid = row.get('COD_id')
    if codid:
        # Monkey patch to make a link
        for tmprow in rows:
            href = ('<a href="http://www.crystallography.net/cod/' +
                    '{id}.html">{id}</a>'.format(id=codid))
            if 'COD' in tmprow[0]:
                tmprow[1] = href
    dynstab = row.get('dynamic_stability_level')
    if dynstab:
        high = 'Min. Hessian eig. > -0.01 meV/Ang^2 AND elastic const. > 0'
        medium = 'Min. Hessian eig. > -2 eV/Ang^2 AND elastic const. > 0'
        low = 'Min. Hessian eig.  < -2 eV/Ang^2 OR elastic const. < 0'
        rows.append([
            'Dynamic stability',
            '<a href="#" data-toggle="tooltip" data-html="true" ' +
            'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'.format(
                low, medium, high, stabilities[dynstab].upper())
        ])

    thermostab = row.get('thermodynamic_stability_level')
    if thermostab:
        high = 'Heat of formation < convex hull + 0.2 eV/atom'
        medium = 'Heat of formation < 0.2 eV/atom'
        low = 'Heat of formation > 0.2 eV/atom'
        rows.append([
            'Thermodynamic stability',
            '<a href="#" data-toggle="tooltip" data-html="true" ' +
            'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'.format(
                low, medium, high, stabilities[thermostab].upper())
        ])

    doi = row.get('monolayer_doi')
    if doi:
        rows.append([
            'Monolayer DOI',
            '<a href="https://doi.org/{doi}" target="_blank">{doi}'
            '</a>'.format(doi=doi)
        ])

    panel = ('Basic properties', [[basictable, UNITCELL], [ATOMS]])
    things = ()
    return panel, things


group = 'Property'
order = 1

if __name__ == '__main__':
    main()
