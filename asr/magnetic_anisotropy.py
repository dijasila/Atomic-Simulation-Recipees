"""Magnetic anisotropy."""
from asr.core import command, read_json, ASRResult, prepare_result
from asr.database.browser import (
    table, make_panel_description, describe_entry, href)
from math import pi


# We don't have mathjax I think, so we should probably use either html
# or unicode.  But z does not exist as unicode superscript, so we mostly
# use html for sub/superscripts.
def equation():
    i = '<sub>i</sub>'
    j = '<sub>j</sub>'
    z = '<sup>z</sup>'
    return (f'E{i} = '
            f'−1/2 J ∑{j} S{i} S{j} '
            f'− 1/2 B ∑{j} S{i}{z} S{j}{z} '
            f'− A S{i}{z} S{i}{z}')


# This panel description actually assumes that we also have results for the
# exchange recipe.

panel_description = make_panel_description(
    """
Heisenberg parameters, magnetic anisotropy and local magnetic
moments. The Heisenberg parameters were calculated assuming that the
magnetic energy of atom i can be represented as

  {equation},

where J is the exchange coupling, B is anisotropic exchange, A is
single-ion anisotropy and the sums run over nearest neighbours. The
magnetic anisotropy was obtained from non-selfconsistent spin-orbit
calculations where the exchange-correlation magnetic field from a
scalar calculation was aligned with the x, y and z directions.

""".format(equation=equation()),
    articles=[
        'C2DB',
        href("""D. Torelli et al. High throughput computational screening for 2D
ferromagnetic materials: the critical role of anisotropy and local
correlations, 2D Mater. 6 045018 (2019)""",
             'https://doi.org/10.1088/2053-1583/ab2c43'),
    ],
)


def get_spin_axis():
    anis = read_json('results-asr.magnetic_anisotropy.json')
    return anis['theta'] * 180 / pi, anis['phi'] * 180 / pi


def get_spin_index():
    anis = read_json('results-asr.magnetic_anisotropy.json')
    axis = anis['spin_axis']
    if axis == 'z':
        index = 2
    elif axis == 'y':
        index = 1
    else:
        index = 0
    return index


def spin_axis(theta, phi):
    import numpy as np
    if theta == 0:
        return 'z'
    elif np.allclose(phi, 90):
        return 'y'
    else:
        return 'x'


def webpanel(result, row, key_descriptions):
    if row.get('magstate', 'NM') == 'NM':
        return []

    magtable = table(row, 'Property',
                     ['magstate', 'magmom',
                      'dE_zx', 'dE_zy'], kd=key_descriptions)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)

    panel = {'title':
             describe_entry(
                 f'Basic magnetic properties ({xcname})',
                 panel_description),
             'columns': [[magtable], []],
             'sort': 11}
    return [panel]


params = '''asr.gs@calculate:calculator +{'mode':'lcao','kpts':(2,2,2)}'''
tests = [{'cli': ['ase build -x hcp Co structure.json',
                  f'asr run "setup.params {params}"',
                  'asr run asr.magnetic_anisotropy',
                  'asr run database.fromtree',
                  'asr run "database.browser --only-figures"']}]


@prepare_result
class Result(ASRResult):

    spin_axis: str
    E_x: float
    E_y: float
    E_z: float
    theta: float
    phi: float
    dE_zx: float
    dE_zy: float

    key_descriptions = {
        "spin_axis": "Magnetic easy axis",
        "E_x": "Soc. total energy, x-direction [meV/unit cell]",
        "E_y": "Soc. total energy, y-direction [meV/unit cell]",
        "E_z": "Soc. total energy, z-direction [meV/unit cell]",
        "theta": "Easy axis, polar coordinates, theta [radians]",
        "phi": "Easy axis, polar coordinates, phi [radians]",
        "dE_zx":
        "Magnetic anisotropy energy between x and z axis [meV/unit cell]",
        "dE_zy":
        "Magnetic anisotropy energy between y and z axis [meV/unit cell]"
    }

    formats = {"ase_webpanel": webpanel}


@command('asr.magnetic_anisotropy',
         tests=tests,
         returns=Result,
         dependencies=['asr.gs@calculate', 'asr.magstate'])
def main() -> Result:
    """Calculate the magnetic anisotropy.

    Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Returns
    -------
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
    """
    from asr.core import read_json
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from gpaw import GPAW

    magstateresults = read_json('results-asr.magstate.json')
    magstate = magstateresults['magstate']

    # Figure out if material is magnetic
    results = {}

    if magstate == 'NM':
        results['E_x'] = 0
        results['E_y'] = 0
        results['E_z'] = 0
        results['dE_zx'] = 0
        results['dE_zy'] = 0
        results['theta'] = 0
        results['phi'] = 0
        results['spin_axis'] = 'z'
        return Result(data=results)

    calc = GPAW('gs.gpw')
    width = 0.001
    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})
    Ex, Ey, Ez = (soc_eigenstates(calc,
                                  theta=theta, phi=phi,
                                  occcalc=occcalc).calculate_band_energy()
                  for theta, phi in [(90, 0), (90, 90), (0, 0)])

    dE_zx = Ez - Ex
    dE_zy = Ez - Ey

    DE = max(dE_zx, dE_zy)
    theta = 0
    phi = 0
    if DE > 0:
        theta = 90
        if dE_zy > dE_zx:
            phi = 90

    axis = spin_axis(theta, phi)

    results.update({'spin_axis': axis,
                    'theta': theta / 180 * pi,
                    'phi': phi / 180 * pi,
                    'E_x': Ex * 1e3,
                    'E_y': Ey * 1e3,
                    'E_z': Ez * 1e3,
                    'dE_zx': dE_zx * 1e3,
                    'dE_zy': dE_zy * 1e3})
    return Result(data=results)


if __name__ == '__main__':
    main.cli()
