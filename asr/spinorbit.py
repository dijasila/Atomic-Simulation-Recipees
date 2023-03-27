from asr.core import command, option, ASRResult, prepare_result
import numpy as np


def sphere_points(N=None, d=None):
    # Calculates N equidistant point on a sphere or
    # finds N from density d in inverse angles
    import math
    if N is None and d is not None:
        N = math.ceil(129600 / (math.pi) * 1 / d**2)

    A = 4 * math.pi
    a = A / N
    d = math.sqrt(a)
    Mtheta = round(math.pi / d)
    dtheta = math.pi / Mtheta
    dphi = a / dtheta
    points = []
    for m in range(Mtheta):
        theta = math.pi * (m + 0.5) / Mtheta
        Mphi = round(2 * math.pi * math.sin(theta) / dphi)
        for n in range(Mphi):
            phi = 2 * math.pi * n / Mphi
            points.append([theta * 180 / math.pi, phi * 180 / math.pi])
    thetas, phis = np.array(points).T
    if 90. not in thetas:
        print('Warning, xy-plane not included in sampling')
    return thetas, phis


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig
    spiraltable = table(row=result, title='Property', keys=['angle_min'],
                        kd=key_descriptions)

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
    key_descriptions = {"soc": "q-constant Spin Orbit correction",
                        "theta": "Angles from z->x",
                        "phi": "Angles from x->y",
                        "angle_min": "Theta, phi angles at minimum",
                        "angle_q": "Theta, phi angles of q-vector"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.spinorbit',
         returns=Result)
@option('--calctxt', help='gpw restart filename', type=str)
@option('--socdensity',
        help='Density of spin orbit energies on the sphere in per angle')
@option('--projected', type=bool,
        help='Boolean to choose projected spin orbit operator')
def main(calctxt: str = "gsq.gpw", socdensity: int = 10,
         projected: bool = True) -> Result:
    from gpaw.spinorbit import soc_eigenstates
    from gpaw import GPAW

    calc = GPAW(calctxt)
    try:
        qn = calc.parameters['mode']['qspiral']
    except:
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
    soc_i = np.array([])
    for i, (theta, phi) in enumerate(zip(thetas, phis)):
        theta += theta_q
        phi += phi_q
        en_soc = soc_eigenstates(calc, projected=projected,
                                 theta=theta, phi=phi).calculate_band_energy()
        en_soc_0 = soc_eigenstates(calc, projected=projected, scale=0.0,
                                   theta=theta, phi=phi).calculate_band_energy()

        soc_i = np.append(soc_i, en_soc - en_soc_0)

    imin = np.argmin(soc_i)
    angle_min = (thetas[imin] + theta_q, phis[imin] + phi_q)
    angle_q = [theta_q, phi_q]
    return Result.fromdata(soc=soc_i, theta=thetas + theta_q, phi=phis + phi_q,
                           angle_min=angle_min, angle_q=angle_q)


def plot_stereographic_energies(row, fname):
    from matplotlib.colors import Normalize
    from matplotlib import pyplot as plt
    from scipy.interpolate import griddata
    from asr.core import read_json

    def stereo_project_point(inpoint, axis=0, r=1, max_norm=1):
        point = np.divide(inpoint * r, inpoint[axis] + r)
        point[axis] = 0
        norm = np.linalg.norm(point)
        if norm >= max_norm:
            point = np.divide(inpoint * r, -inpoint[axis] + r)
            point[axis] = 0
            return ([None, None, None], point + [2, 0, 0])
        return (point, [None, None, None])

    socdata = row.data.get('results-asr.spinorbit.json')
    soc = (socdata['soc'] - min(socdata['soc'])) * 10**6
    theta, phi = socdata['theta'], socdata['phi']
    theta = theta * np.pi / 180
    phi = phi * np.pi / 180
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    upper_points = []
    lower_points = []
    for i in range(len(x)):
        p = [x[i], y[i], z[i]]
        pupper, plower = stereo_project_point(p, axis=2)
        upper_points.append(pupper)
        lower_points.append(plower)

    fig, ax = plt.subplots(1, 1, figsize=(2.2*5, 5))
    norm = Normalize(vmin=min(soc), vmax=max(soc))
    for points in [upper_points, lower_points]:
        X, Y, Z = np.array(points).T
        mask = X != None
        if sum(mask) == 0:
            continue
        X = X[mask]
        Y = Y[mask]
        soci = soc[mask]
        xi = np.linspace(min(X), max(X), 100)
        yi = np.linspace(min(Y), max(Y), 100)
        zi = griddata((X, Y), soci, (xi[None, :], yi[:, None]))
        CS = ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k', norm=norm)
        CS = ax.contourf(xi, yi, zi, 15, cmap=plt.cm.jet, norm=norm)
        #ax.scatter(X, Y, marker='o', c='k', s=5)

    ax.axis('equal')
    ax.set_xlim(-1,3)
    ax.set_ylim(-1,1)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax) # draw colorbar
    cbar.ax.set_ylabel(r'$E_{soc} [\mu eV]$')
    plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
