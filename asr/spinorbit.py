from asr.core import command, option, ASRResult, prepare_result
from typing import List
from itertools import product
import numpy as np


def sphere_points(N=None, d=None):
    # Input:
    #   Option 1: Number of points
    #   Option 2: Density of points
    # Output:
    #   Equidistant points on a (upper half) sphere
    #   (thetas, phis) are list of angles in degrees

    import math
    if N is None and d is not None:
        N = math.ceil(129600 / (math.pi) * 1 / d**2)

    A = 4 * math.pi
    a = A / N
    d = math.sqrt(a)

    # Even number of theta angles ensure 90 deg is included
    Mtheta = round(math.pi / (2 * d)) * 2
    dtheta = math.pi / Mtheta
    dphi = a / dtheta
    points = []

    # Limit theta loop to upper half-sphere
    for m in range(Mtheta // 2 + 1):
        # m = 0 ensure 0 deg is included, Mphi = 1 is used in this case
        theta = math.pi * m / Mtheta
        Mphi = max(round(2 * math.pi * math.sin(theta) / dphi), 1)
        for n in range(Mphi):
            phi = 2 * math.pi * n / Mphi
            points.append([theta, phi])
    thetas, phis = np.array(points).T
    if np.pi / 2 not in thetas:
        print('Warning, xy-plane not included in sampling')
    return thetas * 180 / math.pi, phis * 180 / math.pi % 180


def to_spherical(q):
    # Output: Rad
    q /= np.linalg.norm(q)
    R = np.linalg.norm([q[0], q[1]])
    phi_q = np.arctan2(q[1], q[0])
    r = np.linalg.norm([q[2], R])
    theta_q = np.arctan2(R, q[2])
    assert np.isclose(r, 1.), f'r == {r}'
    return theta_q, phi_q


@prepare_result
class PreResult(ASRResult):
    soc_tp: np.ndarray
    theta_tp: np.ndarray
    phi_tp: np.ndarray
    angle_q: List[float]
    projected_soc: bool
    key_descriptions = {'soc_tp': 'Spin Orbit correction [eV]',
                        'theta_tp': 'Orientation of magnetic order from z->x [deg]',
                        'phi_tp': 'Orientation of magnetic order from x->y [deg]',
                        'angle_q': 'Propagation direction of magnetic order',
                        'projected_soc': 'Projected SOC for spin spirals'}

@command(module='asr.spinorbit',
         resources='1:4h')
@option('--gpwcalc', help='gpw restart filename', type=str)
@option('--soc_density', type=float,
        help='Density of spin orbit energies on the sphere in per angle')
@option('--projected_soc', type=bool,
        help='Boolean to choose projected spin orbit operator')
@option('--width', type=float,
        help='The fermi smearing of the SOC calculation')
def calculate(gpwcalc, soc_density: float = 2.0,
              projected_soc: bool = None, width: float = 0.001) -> ASRResult:
    '''Calculates the spin-orbit coupling at various sampling points on a unit sphere.

    Args:
        gpwcalc (str): The file path for the GPAW calculator context.
        soc_density (int, optional): The density of sampling points on a unit sphere.
        projected (bool, optional): Whether spin-orbit coupling is projected or total.
        width (float, optional): The fermi smearing of soc calculation in eV

    Returns:
        Result: A `Result` object containing the following attributes:
            - `soc`(ndarray): Spin-orbit coupling (eV) at each sampling point.
            - `theta`(ndarray): Polar angle (degrees) at each sampling point.
            - `phi`(ndarray): Azimuthal angle (degrees) at each sampling point.
            - The input Args, except gpwcalc
    '''
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from gpaw import GPAW
    from ase.dft.kpoints import kpoint_convert

    calc = GPAW(gpwcalc)
    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})

    try:
        qn = [round(qi, 15) for qi in calc.parameters['mode']['qspiral']]
        is_collinear = tuple(qn) in list(product([0., 0.5, -0.5], repeat=3))
    except KeyError:
        is_collinear = True
    projected_soc = not is_collinear if projected_soc is None else projected_soc

    if is_collinear:
        theta_q, phi_q = (0, 0)
    else:
        qv = kpoint_convert(cell_cv=calc.atoms.cell, skpts_kc=qn)
        theta_q, phi_q = to_spherical(q=qv)

    theta_tp, phi_tp = sphere_points(d=soc_density)
    phi_tp += phi_q
    # theta_tp += theta_q

    soc_tp = np.array([])
    for theta, phi in zip(theta_tp, phi_tp):
        en_soc = soc_eigenstates(calc=gpwcalc, projected=projected_soc,
                                 theta=theta, phi=phi,
                                 occcalc=occcalc).calculate_band_energy()
        # Noise should not be an issue since it is the same for the calculator
        # en_soc_0 = soc_eigenstates(calc, projected=projected, scale=0.0,
        #                            theta=theta, phi=phi).calculate_band_energy()
        soc_tp = np.append(soc_tp, en_soc)  # - en_soc_0)

    angle_q = [theta_q, phi_q]
    return PreResult.fromdata(soc_tp=soc_tp, theta_tp=theta_tp, phi_tp=phi_tp,
                              angle_q=angle_q, projected_soc=projected_soc)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig
    rows = [['Spinorbit bandwidth', str(np.round(result.get('soc_bw'), 1))],
            ['Spinorbit Minimum (&theta;, &phi;)', '('
             + str(np.round(result.get('theta_min'), 1))
             + ', ' + str(np.round(result.get('phi_min')[1], 1)) + ')']]
    spiraltable = {'type': 'table',
                   'header': ['Property', 'Value'],
                   'rows': rows}

    panel = {'title': 'Spin spirals',
             'columns': [[fig('spinorbit.png')], [spiraltable]],
             'plot_descriptions': [{'function': plot_stereographic_energies,
                                    'filenames': ['spinorbit.png']}],
             'sort': 1}
    return [panel]


@prepare_result
class Result(ASRResult):
    soc_bw: float
    theta_min: float
    phi_min: float
    angle_q: List[float]
    projected_soc: bool
    key_descriptions = {'soc_bw': 'Bandwidth of SOC energies [meV]',
                        'theta_min': 'Angles from z->x [deg]',
                        'phi_min': 'Angles from x->y [deg]',
                        'angle_q': 'Orientation of Qmin [deg]',
                        'projected_soc': 'Projected SOC for spin spirals'}
    formats = {'ase_webpanel': webpanel}


@command(module='asr.spinorbit',
         dependencies=['asr.spinorbit@calculate'],
         resources='1:1m',
         returns=Result)
def main() -> Result:
    from asr.core import read_json

    results = read_json('results-asr.spinorbit@calculate.json')
    soc_tp = results['soc_tp']
    theta_tp = results['theta_tp']
    phi_tp = results['phi_tp']
    angle_q = results['angle_q']
    projected_soc = results['projected_soc']

    tp_min = np.argmin(soc_tp)
    theta_min = theta_tp[tp_min]
    phi_min = phi_tp[tp_min]
    soc_bw = 1e3 * (np.max(soc_tp) - np.min(soc_tp))

    return Result.fromdata(soc_bw=soc_bw, theta_min=theta_min, phi_min=phi_min,
                           angle_q=angle_q, projected_soc=projected_soc)


def plot_stereographic_energies(row, fname, display_sampling=False):
    from matplotlib.colors import Normalize
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from scipy.interpolate import griddata

    def stereo_project_point(inpoint, axis=0, r=1, max_norm=1):
        point = np.divide(inpoint * r, inpoint[axis] + r)
        point[axis] = 0
        return point

    socdata = row.data.get('results-asr.spinorbit.json')
    soc = (socdata['soc'] - min(socdata['soc'])) * 10**3
    theta, phi = socdata['theta'], socdata['phi']
    theta = theta * np.pi / 180
    phi = phi * np.pi / 180
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.array([x, y, z]).T
    projected_points = []
    for p in points:
        projected_points.append(stereo_project_point(p, axis=2))

    plt.figure(figsize=(5 * 1.25, 5))
    ax = plt.gca()
    norm = Normalize(vmin=min(soc), vmax=max(soc))

    X, Y, Z = np.array(projected_points).T

    xi = np.linspace(min(X), max(X), 100)
    yi = np.linspace(min(Y), max(Y), 100)
    zi = griddata((X, Y), soc, (xi[None, :], yi[:, None]))
    ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k', norm=norm)
    ax.contourf(xi, yi, zi, 15, cmap=plt.cm.jet, norm=norm)
    if display_sampling:
        ax.scatter(X, Y, marker='o', c='k', s=5)

    ax.axis('equal')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
    cbar.ax.set_ylabel(r'$E_{soc} [meV]$')
    plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
