"""Plasma frequency."""
from asr.core import command, read_json
from asr.result.resultdata import PlasmaResult


# XXX Why is this even a separate file from polarizability? it doesn't do
# any extra calculation only extra post-processing, which could be a flag
# instead of a whole separate file.
@command('asr.plasmafrequency',
         returns=PlasmaResult,
         dependencies=['asr.polarizability'])
def main() -> PlasmaResult:
    """Calculate polarizability."""
    from ase.io import read
    import numpy as np
    from ase.units import Hartree, Bohr

    atoms = read('structure.json')
    nd = sum(atoms.pbc)
    if not nd == 2:
        raise AssertionError('Plasmafrequency recipe only implemented for 2D')

    # The plasmafrequency has already been calculated in the polarizability recipe
    polresult = read_json("results-asr.polarizability.json")
    plasmafreq_vv = polresult["plasmafreq_vv"].real

    data = {'plasmafreq_vv': plasmafreq_vv}

    if nd == 2:
        wp2_v = np.linalg.eigvalsh(plasmafreq_vv[:2, :2])
        L = atoms.cell[2, 2] / Bohr
        plasmafreq_v = (np.sqrt(wp2_v * L / 2) * Hartree * Bohr**0.5)
        data['plasmafrequency_x'] = plasmafreq_v[0].real
        data['plasmafrequency_y'] = plasmafreq_v[1].real

    return data


if __name__ == '__main__':
    main.cli()
