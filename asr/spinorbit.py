from asr.core import command, option, ASRResult, prepare_result
import numpy as np


def sphere_points(distance=None):
    '''Calculates equidistant points on the upper half sphere

    Returns list of spherical coordinates (thetas, phis) in degrees
    '''

    import math
    N = math.ceil(129600 / (math.pi) * 1 / distance**2)
    if N <= 1:
        return np.array([0.]), np.array([0.])

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

    if not any(thetas - np.pi / 2 < 1e-14):
        import warnings
        warnings.warn('xy-plane not included in sampling')

    return thetas * 180 / math.pi, phis * 180 / math.pi


@prepare_result
class PreResult(ASRResult):
    soc_tp: np.ndarray
    theta_tp: np.ndarray
    phi_tp: np.ndarray
    projected_soc: bool
    key_descriptions = {'soc_tp': 'Spin Orbit correction [eV]',
                        'theta_tp': 'Orientation of magnetic order from z->x [deg]',
                        'phi_tp': 'Orientation of magnetic order from x->y [deg]',
                        'projected_soc': 'Projected SOC for non-collinear spin spirals'}

    def get_projected_points(self):

        def stereo_project_point(inpoint, axis=0, r=1, max_norm=1):
            point = np.divide(inpoint * r, inpoint[axis] + r)
            point[axis] = 0
            return point

        theta, phi = self['theta_tp'], self['phi_tp']
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.array([x, y, z]).T
        projected_points = []
        for p in points:
            projected_points.append(stereo_project_point(p, axis=2))

        return projected_points


@command(module='asr.spinorbit',
         resources='1:4h')
@option('--gpw', help='The file path for the GPAW calculator context.', type=str)
@option('--distance', type=float,
        help='Distance between sample points on the sphere')
@option('--projected', type=bool,
        help='For non-collinear spin spirals, projected SOC should be applied (True)')
@option('--width', type=float,
        help='The fermi smearing of the SOC calculation (eV)')
def calculate(gpw: str = 'gs.gpw', distance: float = 10.0,
              projected: bool = None, width: float = 0.001) -> ASRResult:
    '''Calculates the spin-orbit coupling at equidistant points on a unit sphere. '''
    from gpaw import GPAW
    calc = GPAW(gpw)
    return _calculate(calc, distance, projected, width)


def _calculate(calc, distance: float,
               projected: bool, width: float) -> ASRResult:

    from gpaw.occupations import create_occ_calc
    from gpaw.spinorbit import soc_eigenstates

    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})

    is_collinear = 'qspiral' not in calc.parameters['mode'].keys()
    if not is_collinear:
        qn = calc.parameters['mode']['qspiral']
        assert qn is not None, "qspiral must be a List or similar iterable"
        assert len(qn) == 3, f"The length of qspiral must be 3, we found {len(qn)}"
        if np.linalg.norm(qn) < 1e-14:
            is_collinear = True

    projected = not is_collinear if projected is None else projected
    theta_tp, phi_tp = sphere_points(distance=distance)

    soc_tp = np.array([])
    for theta, phi in zip(theta_tp, phi_tp):

        # The recipe puts magmoms of ferromagnet in x-direction. The rotation angles
        # should reflect this.
        if not projected:
            theta -= 90
        en_soc = soc_eigenstates(calc=calc, projected=projected, theta=theta, phi=phi,
                                 occcalc=occcalc).calculate_band_energy()
        soc_tp = np.append(soc_tp, en_soc)

    return PreResult.fromdata(soc_tp=soc_tp, theta_tp=theta_tp, phi_tp=phi_tp,
                              projected_soc=projected)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig
    rows = [['Spinorbit bandwidth', str(np.round(result.get('soc_bw'), 1))],
            ['Spinorbit Minimum (&theta;, &phi;)', '('
             + str(np.round(result.get('theta_min'), 1))
             + ', ' + str(np.round(result.get('phi_min'), 1)) + ')']]
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
    projected_soc: bool
    key_descriptions = {'soc_bw': 'Bandwidth of SOC energies [meV]',
                        'theta_min': 'Angles from z->x [deg]',
                        'phi_min': 'Angles from x->y [deg]',
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
    projected_soc = results['projected_soc']

    tp_min = np.argmin(soc_tp)
    theta_min = theta_tp[tp_min]
    phi_min = phi_tp[tp_min]
    soc_bw = 1e3 * (np.max(soc_tp) - np.min(soc_tp))

    return Result.fromdata(soc_bw=soc_bw, theta_min=theta_min, phi_min=phi_min,
                           projected_soc=projected_soc)


def plot_stereographic_energies(row, fname, display_sampling=False):
    results = row.data.get('results-asr.spinorbit@calculate.json')
    soc = (results['soc_tp'] - min(results['soc_tp'])) * 10**3
    projected_points = results.get_projected_points()
    _plot_stereographic_energies(projected_points, soc,
                                 fname, display_sampling)


def _plot_stereographic_energies(projected_points, soc,
                                 fname, display_sampling=False):
    from matplotlib.colors import Normalize
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from scipy.interpolate import griddata

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
