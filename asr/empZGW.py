from asr.core import command, option, read_json
import numpy as np


@command('asr.empZGW',
         requires=['results-asr.gw@gw.json'],
         dependencies=['asr.gw@gw'])
@option('-z', '--empz', type=float, default=0.75,
        help='Replacement Z for unphysical Zs')
def main(empz):
    """Implements the empirical-Z method.

    Implements the method described in https://arxiv.org/abs/2009.00314.
    
    This method consists of replacing the G0W0 Z-value with the empirical
    mean of Z-values (calculated from C2DB GW calculations) whenever the
    G0W0 is "quasiparticle-inconsistent", i.e. the G0W0 Z is outside the
    interval [0.5, 1.0]. The empirical mean Z was found to be

    Z0 = 0.75.

    Pseudocode:
    For all states:
        if Z not in [0.5, 1.0]:
            set GW energy = E_KS + Z0 * (Sigma_GW - vxc + exx)

    The last line can be implemented as

    new GW energy = E_KS + (Old GW - E_KS) * Z0 / Z
    """
    Z0 = empz
    
    gwresults = read_json('results-asr.gw@gw.json')
    Z_skn = gwresults['Z']
    e_skn = gwresults['eps']  # Units are eV
    qp_skn = gwresults['qp']  # Units are eV

    indices = np.logical_not(np.logical_and(Z_skn >= 0.5, Z_skn <= 1.0))
    qp_skn[indices] = e_skn[indices] + (qp_skn[indices] - e_skn[indices]) * Z0 / Z_skn[indices]

    results = {}
    results['empZqp'] = qp_skn

    return results
