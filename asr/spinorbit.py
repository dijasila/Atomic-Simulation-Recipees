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
    spiraltable = table(row, 'Property', ['angle_min'], key_descriptions)

    panel = {'title': 'Spin spirals',
             'columns': [[fig('soc_spiral_rot.png')], [spiraltable]],
             'plot_descriptions': [{'function': plot_bandstructure,
                                    'filenames': ['soc_spiral_rot.png']}],
             'sort': 3}
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


def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    socdata = row.data.get('results-asr.soctheta.json')
    soc = socdata['soc'] * 1000
    _, phi = socdata['theta'], socdata['phi']
    i = np.arange(len(phi))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(i, soc, 'k-')
    ax.set_ylabel(r'$E_{soc}$ [meV]')
    ax.set_xlabel('theta, phi')
    ax.set_xticks([0, 90, 180, 270])
    ax.set_xticklabels([r'IP$\parallel$', 'Screw', 'OoP', r'IP$\perp$'])
    ax.set_xlim([min(i), max(i)])
    ax.axvline(90, c='grey', ls='--')
    ax.axvline(180, c='grey', ls='--')
    ax.axhline(soc[0], c='k', ls='--')
    plt.tight_layout()
    plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
