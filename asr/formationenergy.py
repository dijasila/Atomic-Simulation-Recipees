def formationenergy(kvp, data, atoms, verbose=False):
    from asr.references import formation_energy
    kvp['hform'] = formation_energy(atoms) / len(atoms)
    if verbose:
        print('Heat form:', kvp['hform'])


group = 'Property'
