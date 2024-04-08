import numpy as np

def get_spin_gap(direct, calc):
    # Apperently HSE gets bandgap in another way than gs...
    from asr.utils import fermi_level
    from asr.core import read_json
    from ase.dft.bandgap import bandgap
    results_calc = read_json('results-asr.hse@calculate.json')
    eps_skn = results_calc['hse_eigenvalues']['e_hse_skn']
    efermi_nosoc = fermi_level(calc, eigenvalues=eps_skn,
                               nspins=eps_skn.shape[0])
    g0, _, _ =  bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                            direct=direct, output=None, spin=0)

    g1, _, _ =  bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                            direct=direct, output=None, spin=1)
    if g0 == 0.0:
        g = g1
    elif g1 == 0.0:
        g = g0
    else:
        raise ValueError('Both spin channels have a band gap. This is not a half-metallic system.')
    return g


def main():
    """Extract derived quantities from groundstate in hse_nowfs.gpw."""
    from ase.io import read
    from asr.calculators import get_calculator
    import os
    import json

    if not os.path.isfile('structure.json'):
        raise FileNotFoundError("File 'structure.json' does not exist.")

    if not os.path.isfile('hse_nowfs.gpw'):
        raise FileNotFoundError("File 'hse_nowfs.gpw' does not exist.")

    calc = get_calculator()('hse_nowfs.gpw')

    halfmetal_gap = get_spin_gap(direct=False, calc=calc)
    halfmetal_gap_dir = get_spin_gap(direct=True, calc=calc)

    results = {
        'halfmetal_gap_hse': halfmetal_gap,
        'halfmetal_gap_dir_hse': halfmetal_gap_dir,
        }

    with open('halfmetal_hse.json', 'w') as f:
        json.dump(results, f)

    return results

if __name__ == "__main__":
    main()