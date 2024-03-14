"""Plasma frequency."""
from asr.core import command, ASRResult, prepare_result, read_json
import typing


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table

    if row.get('gap', 1) > 0.01:
        return []

    plasmatable = table(row, 'Property', [
        'plasmafrequency_x', 'plasmafrequency_y'], key_descriptions)

    panel = {'title': 'Optical polarizability',
             'columns': [[], [plasmatable]]}
    return [panel]


@prepare_result
class Result(ASRResult):

    plasmafreq_vv: typing.List[typing.List[float]]
    plasmafrequency_x: float
    plasmafrequency_y: float

    key_descriptions = {
        "plasmafreq_vv": "Plasma frequency tensor [Hartree]",
        "plasmafrequency_x": "KVP: 2D plasma frequency (x)"
        "[`eV Å^0.5`]",
        "plasmafrequency_y": "KVP: 2D plasma frequency (y)"
        "[`eV Å^0.5`]",
    }
    formats = {"ase_webpanel": webpanel}


@command('asr.plasmafrequency',
         returns=Result,
         dependencies=['asr.polarizability'])
def main() -> Result:
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
