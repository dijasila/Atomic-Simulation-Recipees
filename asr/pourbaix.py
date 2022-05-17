"""Convex hull stability analysis."""
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from itertools import combinations, chain
import functools

import numpy as np

from asr.core import command, option, ASRResult, prepare_result
from asr.database.browser import (
    fig, table, describe_entry, dl, br, make_panel_description
)

from gpaw import GPAW

from ase.atoms import Atoms
from ase.db import connect
from ase.io import read
from ase.phasediagram import Pourbaix, solvated
from ase.db.row import AtomsRow
from ase.formula import Formula
from ase.build import molecule

import requests as rq

#Lookup tables - Start

#PREDEF_CHEMICAL_POTENTIALS = {
#    'O': {
#        'standard_state': 'O2',
#        'energy': -9.14
#    },
#    'H': {
#        'standard_state': 'H2',
#        'energy': -7.46
#    }
#}

PREDEF_CHEMICAL_POTENTIALS = {
    'O': -4.57,
    'H': -3.73
}

SPECIAL_FORMULAE = {
    "LiO": "Li2O2",
    "NaO": "Na2O2",
    "KO": "K2O2",
    "HO": "H2O2",
    "CsO": "Cs2O2",
    "RbO": "Rb2O2",
    "H": "H2",
    "O": "O2",
    "N": "N2",
    "F": "F2",
    "Cl": "Cl2",
    "Br": "Br2",
    "I": "I2",
}
#Lookup tables - End


def formula_unit(atoms: Atoms) -> Atoms:
    """Return the formula unit of an `ase.Atoms` object.

    Returns the formula unit of a `ase.Atoms` object. This will always use
    the empirical formula. The structures given by `SPECIAL_FORMULAE` are
    automatically converted.

    Parameters
    ----------
    atoms : ase.Atoms
       Needs only the symbols to be specified. 

    Returns
    -------
    ase.Atoms
        The formula unit of the input `atoms`.

    Examples
    --------
    >>> a = ase.Atoms('Li6Cl2O2')
    >>> formula_unit(a)
    Atoms(symbols='Li3ClO', pbc=False)
    """
    fu = Atoms(atoms.get_chemical_formula('metal', empirical=True))

    if str(fu.symbols) in SPECIAL_FORMULAE:
        return Atoms(sorted((fu+fu).symbols))
    return fu


def get_chemsys(symbols):
    elements = set(np.unique(symbols))
    elements.update(['H', 'O'])
    chemsys = list(
        chain.from_iterable(
            [combinations(elements, i+1) for i,_ in enumerate(list(elements))]
        )
    )
    return chemsys


def get_fractional_composition(atoms, elements):
    if isinstance(elements, str):
        elements = [elements]
    count = Counter(atoms.get_chemical_symbols())
    tot_elem = sum([count[e] for e in elements])
    return tot_elem / len(atoms)


def heat_of_formation(atoms, energy, refs):
    n_fu = len(atoms) / len(formula_unit(atoms))
    symbols = atoms.get_chemical_symbols()
    hof = energy - sum([refs[s] for s in symbols])
    return hof / n_fu



################# TESTING #######################

class SessionSingleton:
    __instance = None
    @staticmethod
    def get():
        if SessionSingleton.__instance is None:
            SessionSingleton.__instance = SessionSingleton()
        return SessionSingleton.__instance

    @staticmethod
    def get_session():
        return SessionSingleton.get().session

    def __init__(self) -> None:
        if SessionSingleton.__instance is not None:
            raise RuntimeError(
                "This is a Singleton! use .get()"
            )
        self.session = rq.Session()
        SessionSingleton.__instance = self


def get_MP_references(atoms):
    import json
    from typing import Any, Callable, List, Dict
    from collections import defaultdict

    with open('/home/niflheim/steame/utils/materials-project-API-key', 'r') as f:
        key = f.readline().replace('\n', '')
    global API_KEY
    API_KEY = key

    def unity(val):
        """Perform an identity operation on the input, i.e. just return it."""
        return val

    def convert_to_atoms(structure: dict) -> Atoms:
        """Convert a `'structure'` response to an `ase.Atoms` object.
        
        Using the `'structure'` part of a response from the Materials Project,
        extract the necessary information and convert it to an `ase.Atoms`
        object.
    
        Parameters
        ----------
        structure : dict
            The `structure` response from a query that has `'structure'` as a
            value in its `'properties'` list, converted to a `dict`.
    
        Returns
        -------
        ase.Atoms
        """
        cell = structure['lattice']['matrix']
        sites = structure['sites']
        symbols = [s['species'][0]['element'] for s in sites]
        positions = [p['xyz'] for p in sites]
        # The 'magmom' key can be missing.
        magmoms = [m['properties']['magmom']
                   if 'magmom' in m['properties']
                   else None
                   for m in sites]
    
        return Atoms(symbols=symbols, positions=positions, magmoms=magmoms,
                     cell=cell, pbc=True)
    
    
    treater: Dict[str, Callable] = {
        'structure': convert_to_atoms,
        'final_energy': unity,
        'run_type': unity,
        'is_hubbard': unity,
        'task_id': unity,
        'e_above_hull': unity,
        'oxide_type': unity
    }

    def query_materials_project(data: dict, *, allow_empty_response: bool = False
                                ) -> dict:

        def validate_response(response: dict, *, allow_empty_response: bool = False
                              ) -> dict:
            if not response['valid_response']:
                raise ValueError(
                    response['error']
                )
            # Get out, if we don't care about an empty response
            if allow_empty_response:
                return response
        
            if len(response['response']) == 0:
                raise ValueError(
                    f"The supplied criteria {response['criteria']!r} yielded no "
                    f"results in The Materials Project"
                )
            return response

        ses = SessionSingleton.get_session()
        r = ses.post(
            'https://materialsproject.org/rest/v2/query',
            headers={'X-API-KEY': API_KEY},
            data={k: json.dumps(v) for k, v in data.items()}
        )
        return validate_response(r.json(),
                                 allow_empty_response=allow_empty_response)

    responses = []
    for subsys in get_chemsys(['Zn', 'O']):
        query_str = "-".join(sorted(subsys))
        query_data = {
            'criteria': {
                'chemsys': query_str,
            },
            'properties': [
                'structure',
                'e_above_hull',
                'final_energy'
            ]
        }
        responses.append(query_materials_project(query_data, allow_empty_response=True))

    everything = []
    for resp_chemsys in responses:
        resp = resp_chemsys.get('response')
        for r in resp:
            if r['e_above_hull'] == 0:
                everything.append((
                        convert_to_atoms(r['structure']),
                        r['final_energy'],
                        r['e_above_hull']
                        ))

    refs = []
    reference_energies = {}
    for something in everything:
        nspecies = len(np.unique(something[0].get_chemical_symbols()))
        reduced_formula = something[0].get_chemical_formula('metal', empirical=True)

        if nspecies == 1:
            if reduced_formula in PREDEF_CHEMICAL_POTENTIALS:
                reference_energies.update(
                    {reduced_formula: something[1] / len(something[0])}
                )



        OH_content = get_fractional_composition(something[0], ['O', 'H'])
        if OH_content > 0.85 or 1e-4 < OH_content < 0.095:
            continue

        if reduced_formula in SPECIAL_FORMULAE:
            reduced_formula = SPECIAL_FORMULAE[reduced_formula]

        chempot = heat_of_formation(something[0], something[1], reference_energies)
        refs.append((reduced_formula, chempot))

    #print(reference_energies)

    #stuff = []
    #for something in everything:
    #    if something[2] == 0:
    #        stuff.append(something)
    #print(stuff)


    '''
    materials = defaultdict(list)
    for content in responses:
        atoms, ea_hull, energy, *other = content
        # In the case of elemental species, the e_above_hull is
        # unpredictable, so we use the energy per atom instead
        if len({*atoms.symbols}) == 1:
            ea_hull = energy/len(atoms)

        parcel = (atoms, energy, *other)
        keyname = formula_str(atoms)
        materials[keyname].append((parcel, ea_hull))

    return [
        (min(tasks, key=lambda a: a[1])[0]) for tasks in materials.values()
    ]
    '''

################# END TESTING #######################



def get_OQMD_references(atoms, energy, db_name, solv_refs):
    reduced_formula_calc = atoms.get_chemical_formula('metal', empirical=True)
    db = connect(db_name)
    reference_energies = {}
    refs = []
    
    for subsys in get_chemsys(atoms.get_chemical_symbols()):
        nspecies = len(subsys)
        query_str = ",".join(subsys) + f',nspecies={nspecies}'

        for row in db.select(query_str):
            species = row.toatoms()
            reduced_formula = species.get_chemical_formula('metal', empirical=True)

            if nspecies == 1:
                if reduced_formula in PREDEF_CHEMICAL_POTENTIALS:
                    refdata = {reduced_formula: PREDEF_CHEMICAL_POTENTIALS.get(reduced_formula)}
                else:
                    refdata = {reduced_formula: row.energy / row.natoms}
                reference_energies.update(refdata)

            if reduced_formula == reduced_formula_calc:
                continue

            OH_content = get_fractional_composition(species, ['O', 'H'])
            if OH_content > 0.85 or 1e-4 < OH_content < 0.095:
                continue

            if reduced_formula in SPECIAL_FORMULAE:
                reduced_formula = SPECIAL_FORMULAE[reduced_formula]

            chempot = heat_of_formation(species, row.energy, reference_energies)
            refs.append((reduced_formula, chempot))

    refs.append((
        reduced_formula_calc,
        heat_of_formation(
            atoms,
            energy,
            reference_energies)
    ))
    #print(refs)
    #print(reference_energies)

    if solv_refs:
        refs += get_solvated_species(atoms, solv_refs)

    return refs


def get_solvated_species(atoms, source):
    from ase import units
    kcalmol2eV = lambda x: x * units.mol / units.kcal

    if source == 'ASE':
        solv = solvated(atoms.get_chemical_formula())
        #solv_eV = [(name, kcalmol2eV(energy)) for name, energy in solv]
        solv_eV = [(name, energy) for name, energy in solv]
        return solv_eV


@command('asr.pourbaix')
@option('--gpw', type=str)
@option('--dft-refs', type=str)
@option('--solv-refs', type=str)
@option('--urange', type=list)
@option('--phrange', type=list)
@option('--conc', type=float)
@option('--npoints', type=int)
def main(gpw: str='gs.gpw',
         dft_refs: str='OQMD',
         solv_refs: str='ASE',
         phrange: list=[-2, 16],
         urange: list=[-3, 3],
         conc: float=1e-6,
         npoints: int = 600):

    from asr.relax import main as relax
    from asr.gs import main as groundstate
    from asr.core import read_json
    import matplotlib.pyplot as plt

    test = True

    if test:
        atoms = read('structure.json')
        with open('energy.txt', 'r') as f:
            energy = float(f.readline())

    else:
        calc = GPAW(gpw)
        atoms = calc.get_atoms()
        energy = calc.get_potential_energy()

    if dft_refs == 'OQMD':
        refs = get_OQMD_references(
            atoms,
            energy,
            '/home/niflheim/steame/utils/oqmd123.db',
            solv_refs
        )

    if dft_refs == 'MP':
        refs = get_MP_references(
            atoms,
            #energy,
            #'/home/niflheim/steame/utils/oqmd123.db',
            #solv_refs
        )

    def get_atom_count(atoms):
        fu = formula_unit(atoms)
        return Counter(fu.get_chemical_symbols())

    count = get_atom_count(atoms)
    pbd = Pourbaix(refs, **count)

    pH = np.linspace(*phrange, npoints)
    ratio = np.ptp(urange) / np.ptp(phrange)
    U = np.linspace(*urange, int(npoints * ratio))

    pourbaix = pbd.diagram(
            U, pH, 
            conc=conc,
            plot=True,
            show=False,
            savefig='pourbaix.png'
    )
    metastab = pbd.metastability(
            U, pH,
            conc=conc,
            cap = 1.0,
            plot=True,
            show=False,
            normalize=True,
            savefig='metastability.png'
    )

    '''
    TO DO:
    - correct energies with reference oxides
        requires building large lookup table
    - add MP functionality
    - adjust output according to convexhull recipe

    RECIPE DESIGN:
    - include refs from C2DB?
    - which type of input should be obtained automatically/provided by user?
          1) database
          2) (name, energy) tuple list
    '''


if __name__ == '__main__':
    main.cli()
