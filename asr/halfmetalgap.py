import numpy as np

def get_spin_gap(direct, calc):
    from ase.dft.bandgap import bandgap
    g0, _, _ = bandgap(calc, direct=direct, output=None, spin=0)

    g1, _, _ = bandgap(calc, direct=direct, output=None, spin=1)
    
    if g0 == 0.0:
        g = g1
    elif g1 == 0.0:
        g = g0
    else:
        raise ValueError('Both spin channels have a band gap. This is not a half-metallic system.')

    return g

def main():
    """Extract derived quantities from groundstate in gs.gpw."""
    from ase.io import read
    from asr.calculators import get_calculator
    import os
    import json

    if not os.path.isfile('structure.json'):
        raise FileNotFoundError("File 'structure.json' does not exist.")

    if not os.path.isfile('gs.gpw'):
        raise FileNotFoundError("File 'gs.gpw' does not exist.")

    atoms = read('structure.json')
    calc = get_calculator()('gs.gpw')
    pbc = atoms.pbc
    ndim = np.sum(pbc)

    if ndim == 2:
        assert not pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        # For 2D materials we check that the calculater used a dipole
        # correction if the material has an out-of-plane dipole

        # Small hack
        atoms = calc.atoms
        atoms.calc = calc

    halfmetal_gap = get_spin_gap(direct=False, calc=calc)
    halfmetal_gap_dir = get_spin_gap(direct=True, calc=calc)

    results = {
        'halfmetal_gap': halfmetal_gap,
        'halfmetal_gap_dir': halfmetal_gap_dir,
        }

    with open('halfmetal.json', 'w') as f:
        json.dump(results, f)

    return results

if __name__ == "__main__":
    main()