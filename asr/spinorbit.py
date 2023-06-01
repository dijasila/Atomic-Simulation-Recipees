from asr.core import command, option, ASRResult, prepare_result
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
            points.append([theta * 180 / math.pi, phi * 180 / math.pi])
    thetas, phis = np.array(points).T
    if 90. not in thetas:
        print('Warning, xy-plane not included in sampling')
    return thetas, phis


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig
    rows = [['Spinorbit bandwidth', str(np.round(1e3 * (max(result.get('soc'))
                                                  - min(result.get('soc'))), 1))],
            ['Spinorbit Minimum (&theta;, &phi;)', '('
             + str(np.round(result.get('angle_min')[0], 1))
             + ', ' + str(np.round(result.get('angle_min')[1], 1)) + ')']]
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
    soc: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    angle_min: tuple
    angle_q: tuple
    projected: bool
    key_descriptions = {"soc": "q-constant Spin Orbit correction",
                        "theta": "Angles from z->x",
                        "phi": "Angles from x->y",
                        "angle_min": "Theta, phi angles at minimum",
                        "angle_q": "Theta, phi angles of q-vector",
                        "projected": "Boolean indicates projected spin orbit operator"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.spinorbit',
         returns=Result)
@option('--calctxt', help='gpw restart filename', type=str)
@option('--socdensity', type=float,
        help='Density of spin orbit energies on the sphere in per angle')
@option('--projected', type=bool,
        help='Boolean to choose projected spin orbit operator')
def main(calctxt: str = "gsq.gpw", socdensity: float = 10.0,
         projected: bool = True) -> Result:

    '''Calculates the spin-orbit coupling at various sampling points on a unit sphere.

    Args:
        calctxt (str, optional): The file path for the GPAW calculator context.
        socdensity (int, optional): The density of sampling points on a unit sphere.
        projected (bool, optional): Whether spin-orbit coupling is projected or tota´l.

    Returns:
        Result: A `Result` object containing the following attributes:
            - `soc`(ndarray): Spin-orbit coupling (eV) at each sampling point.
            - `theta`(ndarray): Polar angle (degrees) of each sampling point.
            - `phi`(ndarray): Azimuthal angle (degrees) of each sampling point.
            - `angle_min`(list): Polar and azimuthal angles (in radians) at which
                                  the minimum spin-orbit coupling occurs.
            - `angle_q`(list): Polar and azimuthal angles (degrees) of the wavevector q
            - `projected`(bool): Whether spin-orbit coupling is projected or total.
    '''
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from gpaw import GPAW

    calc = GPAW(calctxt)
    width = 0.001
    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})

    try:
        qn = calc.parameters['mode']['qspiral']
    except KeyError:
        qn = 0  # Quick collinear / non-Q implementation

    if not np.isclose(np.linalg.norm(qn), 0):
        qn /= np.linalg.norm(qn)
        if qn[0] < 0:
            qn = -qn
        if qn[0] == 0:
            qn = np.abs(qn)
        sign = np.sign(np.cross([1, 0, 0], qn)[2])

        # Starting phases are in xy-plane pointing parallel to q
        phi_q = np.arccos(np.clip(np.dot(qn, [1, 0, 0]),
                                  -1.0, 1.0)) * 180 / np.pi * sign
        theta_q = np.arccos(np.dot(qn, [0, 0, 1])) * 180 / np.pi - 90
    else:
        phi_q = 0
        theta_q = 0

    thetas, phis = sphere_points(d=socdensity)
    thetas += theta_q
    phis += phi_q

    soc = np.array([])
    for theta, phi in zip(thetas, phis):
        en_soc = soc_eigenstates(calc=calctxt, projected=projected,
                                 theta=theta, phi=phi,
                                 occcalc=occcalc).calculate_band_energy()
        # Noise should not be an issue since it is the same for the calculator
        # en_soc_0 = soc_eigenstates(calc, projected=projected, scale=0.0,
        #                            theta=theta, phi=phi).calculate_band_energy()
        soc = np.append(soc, en_soc)  # - en_soc_0)

    imin = np.argmin(soc)
    angle_min = [thetas[imin], phis[imin]]
    angle_q = [theta_q, phi_q]
    return Result.fromdata(soc=soc, theta=thetas, phi=phis,
                           angle_min=angle_min, angle_q=angle_q, projected=projected)


def plot_stereographic_energies(row, fname, display_sampling=False):
    from matplotlib.colors import Normalize
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

    fig, ax = plt.subplots(1, 1, figsize=(5*1.25, 5))
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
