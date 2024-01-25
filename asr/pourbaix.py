from typing import Union
from ase.pourbaix import Species, Pourbaix, PREDEF_ENERGIES
from ase.phasediagram import solvated
from ase.db import connect


def get_solid_refs(material, db_name, energy_key, predef_energies):
    """Extract solid references in the compositional space
       of a given material from a database"""

    db = connect(db_name)
    element_energies = {}
    refs = {}

    for subsys in material.get_chemsys():
        nspecies = len(subsys)
        query_str = ",".join(subsys) + f',nspecies={nspecies}'

        for row in db.select(query_str):
            energy = row[energy_key]
            ref = Species(row.formula)
            name = ref.name

            if nspecies == 1:
                energy_elem = predef_energies.get(
                    name,
                    energy / row.natoms
                )
                element_energies[name] = energy_elem

            OH_content = ref.get_fractional_composition(['O', 'H'])
            if OH_content > 0.85 or 1e-4 < OH_content < 0.095:
                continue

            chempot = ref.get_formation_energy(energy, element_energies)
            refs[name] = chempot

    return refs, element_energies


def get_solvated_refs(name):
    """Extract ionic and solvated species"""
    ref_dct = {}
    solv = solvated(name)
    for name, energy in solv:
        if name not in ['H+(aq)', 'H2O(aq)']:
            ref_dct[name] = energy
    return ref_dct


def get_references(
        material, db_name,
        computed_energy=None,
        include_aq=True,
        energy_key='energy',
        predef_energies=PREDEF_ENERGIES):

    if predef_energies is None:
        predef_energies = {}

    species = Species(material)
    refs, element_energies = get_solid_refs(
            species, db_name, energy_key, predef_energies
    )
    refs.update(get_solvated_refs(species.name))

    if computed_energy:
        chempot = species.get_formation_energy(
            computed_energy, element_energies
        )
        refs[species.name] = chempot
    else:
        if species.name not in refs.keys():
            raise ValueError(\
                f'Your material has not been found in {db_name}. '
                f'Please provide a total energy for it!')
    return refs, species.name


def main(material: str,
         computed_energy: Union[float, None]=None,
         database: str='oqmd123.db',
         energy_key: str='energy',
         pHrange: Union[list, tuple]=[0, 14],
         Urange: Union[list, tuple]=[-3, 3],
         conc: float=1e-6,
         counter: str='SHE',
         npoints: int=300,
         show: bool=False,
         savefig: str='pourbaix.png'):

    refs, name = get_references(
        material,
        database,
        computed_energy,
        include_aq=True,
        energy_key=energy_key,
        predef_energies=PREDEF_ENERGIES
    )

    pbx = Pourbaix(name, refs, conc=conc, counter=counter)
    pbx.plot(
        Urange, pHrange,
        npoints=npoints, 
        show=show,
        savefig=savefig
    )


if __name__ == '__main__':
    import sys
    if sys.argv[2]:
        energy=float(sys.argv[2])
    else:
        energy=None

    main(sys.argv[1], computed_energy=energy)
