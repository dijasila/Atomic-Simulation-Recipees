"""Magnetic anisotropy."""
import numpy as np
from ase import Atoms
import asr
from asr.core import (command, ASRResult, prepare_result,
                      option, AtomsFile)

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


def spin_axis(theta, phi):
    if theta == 0:
        return 'z'
    elif np.allclose(phi, 90):
        return 'y'
    else:
        return 'x'


def webpanel(result, context):
    magstate = context.magstate().result
    if magstate['magstate'] == 'NM':
        return []

    magtable = table(result, 'Property',
                     ['magstate', 'magmom',
                      'dE_zx', 'dE_zy'], kd=context.descriptions)

    panel = {'title':
             describe_entry(
                 f'Basic magnetic properties ({context.xcname})',
                 panel_description),
             'columns': [[magtable], []],
             'sort': 11}
    return [panel]


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
        "E_x": "Soc. total energy, x-direction [eV/unit cell]",
        "E_y": "Soc. total energy, y-direction [eV/unit cell]",
        "E_z": "Soc. total energy, z-direction [eV/unit cell]",
        "theta": "Easy axis, polar coordinates, theta [radians]",
        "phi": "Easy axis, polar coordinates, phi [radians]",
        "dE_zx":
        "Magnetic anisotropy energy between x and z axis [meV/unit cell]",
        "dE_zy":
        "Magnetic anisotropy energy between y and z axis [meV/unit cell]"
    }

    formats = {"webpanel2": webpanel}

    def spin_index(self):
        axis = self['spin_axis']
        if axis == 'z':
            index = 2
        elif axis == 'y':
            index = 1
        else:
            index = 0
        return index

    def spin_angles(self):
        return self['theta'] * 180 / pi, self['phi'] * 180 / pi


@command('asr.c2db.magnetic_anisotropy')
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
@asr.calcopt
def main(atoms: Atoms,
         calculator: dict = {
             'name': 'gpaw',
             'mode': {'name': 'pw', 'ecut': 800},
             'xc': 'PBE',
             'kpts': {'density': 12.0, 'gamma': True},
             'occupations': {'name': 'fermi-dirac',
                             'width': 0.05},
             'convergence': {'bands': 'CBM+3.0'},
             'nbands': '200%',
             'txt': 'gs.txt',
             'charge': 0
         }) -> Result:
    """Calculate the magnetic anisotropy.

    Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Returns
    -------
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
    """
    from asr.c2db.magstate import main as magstate
    from asr.c2db.gs import calculate as calculategs
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc

    magstateresults = magstate(
        atoms=atoms,
        calculator=calculator)
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

    calculateresult = calculategs(atoms=atoms, calculator=calculator)
    calc = calculateresult.calculation.load()
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
