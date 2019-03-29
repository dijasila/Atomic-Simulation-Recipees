def formationenergy(kvp, data, atoms, verbose=False):
    kvp['hform'] = formation_energy(atoms) / len(atoms)
    if verbose:
        print('Heat form:', kvp['hform'])


group = 'Property'
