"""Aqueous stability through Pourbaix diagrams."""
from itertools import combinations, chain
from ase.phasediagram import Pourbaix, solvated
from ase.formula import Formula
from asr.core import command, option, ASRResult, prepare_result
from asr.database.browser import (
    fig, table, describe_entry, dl, br, make_panel_description
)


PREDEF_ENERGIES = {
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
    "N": "N2",
    "F": "F2",
    "Cl": "Cl2",
    "Br": "Br2",
    "I": "I2",
}

OQMD = '/home/niflheim/steame/utils/oqmd123.db'

class Reference():

    def __init__(self, formula, fmt='metal'):
        if isinstance(formula, str):
            formula = Formula(formula, format=fmt)
        elif isinstance(formula, Formula):
            formula = formula.convert(fmt)
        else:
            raise ValueError("""\
                    formula must be of type str or ase.formula.Formula""")

        self.formula = str(formula)
        self.natoms = len(formula)
        self.count = formula.count()
        reduced, self.n_fu = formula.reduce()
        self.reduced = SPECIAL_FORMULAE.get(
            str(reduced), str(reduced)
        )

    def get_chemsys(self):
        elements = set(self.count.keys())
        elements.update(['H', 'O'])
        chemsys = list(
            chain.from_iterable(
                [combinations(elements, i+1) for i,_ in enumerate(list(elements))]
            )
        )
        return chemsys

    def get_fractional_composition(self, elements):
        N_all = sum(self.count.values())
        N_elem = sum([self.count.get(e, 0) for e in elements])
        return N_elem / N_all

    def get_entry(self, energy, refs):
        elem_energy = sum([refs[s] * n for s, n in self.count.items()])
        hof = (energy - elem_energy) / self.n_fu
        return {self.reduced: hof}


def get_solid_references(material, db_name, computed_energy=None):
    from ase.db import connect
    reference_energies = {}
    refs = {}

    with connect(db_name) as db:
        for subsys in material.get_chemsys():
            nspecies = len(subsys)
            query_str = ",".join(subsys) + f',nspecies={nspecies}'

            for row in db.select(query_str):
                ref = Reference(row.formula)
                energy = row.energy
                print(row.formula, energy)

                if nspecies == 1:
                    name = ref.reduced
                    energy_elem = PREDEF_ENERGIES.get(
                        name,
                        energy / row.natoms
                    )
                    reference_energies[name] = energy_elem

                OH_content = ref.get_fractional_composition(['O', 'H'])
                if OH_content > 0.85 or 1e-4 < OH_content < 0.095:
                    continue

                entry = ref.get_entry(energy, reference_energies)
                refs.update(entry)

    if computed_energy is not None:
        entry = material.get_entry(computed_energy, reference_energies)
        refs.update(entry)

    return refs


def get_references(material, db_name, computed_energy=None):
    if isinstance(material, str):
        material = Reference(material)

    allrefs = get_solid_references(material, db_name, computed_energy)
    solv_refs = solvated(material.formula)
    solv_refs_dct = {name: energy for name, energy in solv_refs}
    allrefs.update(solv_refs_dct)
    refs_list = [(name, energy) for name, energy in allrefs.items()]

    return refs_list


def compute_diagrams(
        name,
        references,
        phrange=[-2, 16],
        urange=[-3, 3],
        conc=1e-6,
        npoints=300):
    import numpy as np

    pH = np.linspace(*phrange, npoints)
    ratio = np.ptp(urange) / np.ptp(phrange)
    U = np.linspace(*urange, int(npoints * ratio))

    pbd = Pourbaix(references, formula=name)
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


@command('asr.pourbaix')
@option('--gpw', type=str)
@option('--database', type=str)
@option('--urange', type=list)
@option('--phrange', type=list)
@option('--conc', type=float)
@option('--npoints', type=int)
def main(gpw: str='gs.gpw',
         database: str=OQMD,
         phrange: list=[-2, 16],
         urange: list=[-3, 3],
         conc: float=1e-6,
         npoints: int = 600):
    from gpaw import GPAW

    calc = GPAW(gpw)
    atoms = calc.get_atoms()
    energy = calc.get_potential_energy()
    raw_formula = atoms.get_chemical_formula()
    material = Reference(raw_formula)

    references = get_references(material, database, energy)
    print(references)

    compute_diagrams(material.reduced, references)

    '''
    TO DO:
    - add Result
    - figure out concentration problem
    - add OH- (?)
    - add MP functionality for solvated references
    - adjust output according to convexhull recipe
    '''


if __name__ == '__main__':
    main.cli()
