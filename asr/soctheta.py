from asr.core import command, option, DictStr, argument, ASRResult, prepare_result
from typing import List, Union
import numpy as np
from os import path

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
    key_descriptions={"soc":"q-constant Spin Orbit correction",
                      "theta":"Angles from z->x",
                      "phi":"Angles from x->y",
                      "angle_min":"Theta, phi angles at minimum"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.soctheta',
         requires=['results-asr.spinspiral.json'],
         returns=Result)
@option('--projected', help='Boolean to choose projected spin orbit operator')
def main(projected: bool = True) -> Result:
    from glob import glob
    from ase.io import read
    from asr.core import read_json
    from gpaw.spinorbit import soc_eigenstates
    from gpaw import GPAW
    atoms = read('structure.json')
    res = read_json('results-asr.spinspiral.json')
    calc = GPAW(f"gsq{np.argmin(res.energies)}.gpw")
    qn = calc.parameters['mode']['qspiral']
    #assert np.isclose(np.linalg.norm(qn), 0) is False
    print(qn)
    qn /= np.linalg.norm(qn)
    if qn[0] < 0:
        qn = -qn
    if qn[0] == 0:
        qn = np.abs(qn)
    sign = np.sign(np.cross([1, 0, 0], qn)[2])
    
    # Starting phases are in xy-plane pointing parallel to q
    phi_q =  np.arccos(np.clip(np.dot(qn, [1, 0, 0]), -1.0, 1.0)) * 180/np.pi * sign
    theta_q = np.arccos(np.dot(qn, [0, 0, 1])) * 180/np.pi - 90
        
    ip2scw_theta = np.arange(0, 90)
    ip2scw_phi = np.zeros_like(ip2scw_theta)
    scw2oop_phi = np.arange(0, 90)
    scw2oop_theta = np.ones_like(scw2oop_phi)*90
    oop2ip_theta = np.arange(90, -1, -1)
    oop2ip_phi = np.ones_like(oop2ip_theta)*90
    
    thetas = np.concatenate((ip2scw_theta, scw2oop_theta, oop2ip_theta), axis=0)
    phis = np.concatenate((ip2scw_phi, scw2oop_phi, oop2ip_phi), axis=0)

    SOC = np.array([])
    THETA = np.array([])
    PHI = np.array([])
    for i, (theta, phi) in enumerate(zip(thetas, phis)):
        theta_i = theta_q + theta
        phi_i = phi_q + phi
        en_soc = soc_eigenstates(calc, projected=projected,
                                 theta=theta_i, phi=phi_i).calculate_band_energy()
        en_soc_0 = soc_eigenstates(calc, projected=projected, 
                                   scale=0.0, theta=theta_i, phi=phi_i).calculate_band_energy()

        SOC = np.append(SOC, en_soc - en_soc_0)
        THETA = np.append(THETA, theta_i)
        PHI = np.append(PHI, phi_i)
    imin = np.argmin(SOC)
    angle_min = (THETA[imin], PHI[imin])
    return Result.fromdata(soc=SOC, theta=THETA, phi=PHI, angle_min=angle_min)

def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    socdata = row.data.get('results-asr.soctheta.json')
    soc = socdata['soc'] * 1000
    theta, phi = socdata['theta'], socdata['phi']
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
