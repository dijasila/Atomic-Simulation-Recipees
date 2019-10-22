from asr.core import command, option, read_json
from click import Choice


# This function is basically doing the exact same as HSE and could
# probably be refactored
def bs_gw(row,
          filename='gw-bs.png',
          figsize=(6.4, 4.8),
          fontsize=10,
          show_legend=True,
          s=0.5):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from ase.dft.kpoints import labels_from_kpts

    data = row.data.get('results-asr.gw.json')
    path = data['bandstructure']['path']
    mpl.rcParams['font.size'] = fontsize
    ef = data['efermi_gw_soc']

    if row.get('evac') is not None:
        label = r'$E - E_\mathrm{vac}$ [eV]'
        reference = row.get('evac')
    else:
        label = r'$E - E_\mathrm{F}$ [eV]'
        reference = ef

    emin = row.get('vbm_gw', ef) - 5 - reference
    emax = row.get('cbm_gw', ef) + 10 - reference

    e_mk = data['bandstructure']['e_int_mk'] - reference
    x, X, labels = labels_from_kpts(path.kpts, row.cell)

    # hse with soc
    style = dict(
        color='k',
        # label='HSE',
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    for e_m in e_mk:
        ax.plot(x, e_m, **style)
    ax.set_ylim([emin, emax])
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylabel(label)
    ax.set_xlabel('$k$-points')
    ax.set_xticks(X)
    ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in labels])

    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    ax.axhline(ef - reference, c='k', ls=':')
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, ef - reference),
        ha='left',
        va='bottom',
        fontsize=fontsize * 1.3)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    # add PBE band structure with soc
    from asr.bandstructure import add_bs_pbe
    if 'results-asr.bandstructure.json' in row.data:
        ax = add_bs_pbe(row, ax, reference=row.get('evac', row.get('efermi')))

    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    ax.plot([], [], **style, label='GW')
    plt.legend(loc='upper right')

    if not show_legend:
        ax.legend_.remove()
    plt.savefig(filename, bbox_inches='tight')


@command(requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         creates=['gs_gw.gpw', 'gs_gw_nowfs.gpw'])
@option('--kptdensity', help='K-point density')
@option('--ecut', help='Plane wave cutoff')
def gs(kptdensity=5.0, ecut=200.0):
    """Calculate GW"""
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW
    import numpy as np

    # check that the system is a semiconductor
    calc = GPAW('gs.gpw', txt=None)
    pbe_gap, _, _ = bandgap(calc, output=None)
    if pbe_gap < 0.05:
        raise Exception("GW: Only for semiconductors, PBE gap = " +
                        str(pbe_gap) + " eV is too small!")

    # check that the system is small enough
    atoms = calc.get_atoms()
    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, " +
                        str(len(atoms)) + " > 4 atoms!")

    # setup k points/parameters
    dim = np.sum(atoms.pbc.tolist())
    if dim == 3:
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
    elif dim == 2:
        def get_kpts_size(atoms, kptdensity):
            """trying to get a reasonable monkhorst size which hits high
            symmetry points
            """
            from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
            size, offset = k2so(atoms=atoms, density=kptdensity)
            size[2] = 1
            for i in range(2):
                if size[i] % 6 != 0:
                    size[i] = 6 * (size[i] // 6 + 1)
            kpts = {'size': size, 'gamma': True}
            return kpts

        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)
    elif dim == 1:
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        # TODO remove unnecessary k
        raise NotImplementedError('asr for dim=1 not implemented!')
    elif dim == 0:
        kpts = {'density': 0.0, 'gamma': True, 'even': True}
        # TODO only Gamma
        raise NotImplementedError('asr for dim=0 not implemented!')

    # we need energies/wavefunctions on the correct grid
    calc = GPAW(
        'gs.gpw',
        txt='gs_gw.txt',
        fixdensity=True,
        kpts=kpts)
    calc.get_potential_energy()
    calc.diagonalize_full_hamiltonian(ecut=ecut)
    calc.write('gs_gw_nowfs.gpw')
    calc.write('gs_gw.gpw', mode='all')


@command(requires=['gs_gw.gpw'],
         dependencies=['asr.gw@gs'])
@option('--ecut', help='Plane wave cutoff')
@option('--mode', help='GW mode',
        type=Choice(['G0W0', 'GWG']))
def gw(ecut=200.0, mode='G0W0'):
    """Calculate GW"""
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW
    from gpaw.response.g0w0 import G0W0
    import numpy as np

    # check that the system is a semiconductor
    calc = GPAW(gs, txt=None)
    pbe_gap, _, _ = bandgap(calc, output=None)
    if pbe_gap < 0.05:
        raise Exception("GW: Only for semiconductors, PBE gap = " +
                        str(pbe_gap) + " eV is too small!")

    # check that the system is small enough
    atoms = calc.get_atoms()
    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, " +
                        str(len(atoms)) + " > 4 atoms!")

    # Setup parameters
    dim = np.sum(atoms.pbc.tolist())
    if dim == 3:
        truncation = 'wigner-seitz'
        q0_correction = False
    elif dim == 2:
        truncation = '2D'
        q0_correction = True
    elif dim == 1:
        raise NotImplementedError('asr for dim=1 not implemented!')
        truncation = '1D'
        q0_correction = False
    elif dim == 0:
        raise NotImplementedError('asr for dim=0 not implemented!')
        truncation = '0D'
        q0_correction = False

    if mode == 'GWG':
        raise NotImplementedError('GW: asr for GWG not implemented!')

    lb, ub = max(calc.wfs.nvalence // 2 - 8, 0), calc.wfs.nvalence // 2 + 4

    calc = G0W0(calc='gs_gw.gpw',
                bands=(lb, ub),
                ecut=ecut,
                ecut_extrapolation=True,
                truncation=truncation,
                nblocksmax=True,
                q0_correction=q0_correction,
                filename='g0w0',
                restartfile='g0w0.tmp',
                savepckl=False)

    results = calc.calculate()
    results['minband'] = lb
    results['maxband'] = ub
    return results


def plot_renorm_factor(row, filename):
    from matplotlib import pyplot as plt
    data = row.data.get('results-asr.gw@gw.json')
    q_skn = data['qp']
    Z_skn = data['Z']
    reference = row.get('evac', row.get('ef'))
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(Z_skn.ravel(), q_skn.ravel() - reference,
                s=2)
    emin = row.get('vbm_gw', row.get('ef')) - 5 - reference
    emax = row.get('cbm_gw', row.get('ef')) + 10 - reference
    if row.get('evac') is not None:
        plt.ylabel(r'$E - E_\mathrm{vac}$ [eV]')
    else:
        plt.ylabel(r'$E - E_\mathrm{F}$ [eV]')
    plt.ylim(emin, emax)
    plt.xlabel('Renormalization factor Z')
    plt.tight_layout()
    plt.savefig(filename)


def webpanel(row, key_descriptions):
    from asr.browser import fig, table

    prop = table(row, 'Property', [
        'gap_gw', 'dir_gap_gw', 'vbm_gw', 'cbm_gw'
    ], key_descriptions)

    panel = {'title': 'Electronic band structure (GW)',
             'columns': [[fig('gw-bs.png'), prop], [fig('renorm.png')]],
             'plot_descriptions': [{'function': bs_gw,
                                    'filenames': ['gw-bs.png']},
                                   {'function': plot_renorm_factor,
                                    'filenames': ['renorm.png']}]}
    return [panel]


@command(requires=['results-asr.gw@gw.json', 'gs_gw_nowfs.gpw'],
         dependencies=['asr.gw@gw', 'asr.gw@gs'],
         webpanel=webpanel)
def main():
    import numpy as np
    from gpaw import GPAW
    from asr.utils.gpw2eigs import fermi_level
    from ase.dft.bandgap import bandgap
    from asr.hse import MP_interpolate
    from types import SimpleNamespace
    
    calc = GPAW('gs_gw_nowfs.gpw', txt=None)
    gwresults = SimpleNamespace(**read_json('results-asr.gw@gw.json'))

    lb = gwresults.minband
    ub = gwresults.maxband

    delta_skn = gwresults.qp - gwresults.eps

    # Interpolate band structure
    results = MP_interpolate(calc, delta_skn, lb, ub)
    kd = {}

    # First get stuff without SOC
    eps_skn = gwresults.qp
    efermi_nosoc = fermi_level(calc, eps_skn=eps_skn,
                               nelectrons=(calc.get_number_of_electrons() -
                                           2 * lb))
    gap, p1, p2 = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                             direct=True, output=None)
    if gap > 0:
        ibzkpts = calc.get_ibz_k_points()
        kvbm_nosoc = ibzkpts[p1[1]]  # k coordinates of vbm
        kcbm_nosoc = ibzkpts[p2[1]]  # k coordinates of cbm
        vbm = eps_skn[p1]
        cbm = eps_skn[p2]
        subresults = {'vbm_gw_nosoc': vbm,
                      'cbm_gw_nosoc': cbm,
                      'dir_gap_gw_nosoc': gapd,
                      'gap_gw_nosoc': gap,
                      'kvbm_nosoc': kvbm_nosoc,
                      'kcbm_nosoc': kcbm_nosoc}

        kd.update({'vbm_gw_nosoc': 'GW valence band max. w/o soc [eV]',
                   'cbm_gw_nosoc': 'GW condution band min. w/o soc [eV]',
                   'dir_gap_gw_nosoc': 'GW direct gap w/o soc [eV]',
                   'gap_gw_nosoc': 'GW gap w/o soc [eV]',
                   'kvbm_nosoc': 'k-point of GW valence band max. w/o soc',
                   'kcbm_nosoc': 'k-point of GW conduction band min. w/o soc'})
        results.update(subresults)

    # Get the SO corrected GW QP energires
    from gpaw.spinorbit import get_spinorbit_eigenvalues as get_soc_eigs
    from asr.utils.gpw2eigs import get_spin_direction
    bandrange = np.arange(lb, ub)
    theta, phi = get_spin_direction()
    e_mk = get_soc_eigs(calc, gw_kn=eps_skn,
                        bands=bandrange,
                        return_spin=False,
                        theta=theta, phi=phi)

    eps = e_mk.transpose()[np.newaxis]  # e_skm, dummy spin index
    efermi_soc = fermi_level(calc, eps_skn=eps,
                             nelectrons=(2 *
                                         (calc.get_number_of_electrons() -
                                          2 * lb)))
    gap, p1, p2 = bandgap(eigenvalues=eps, efermi=efermi_soc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps, efermi=efermi_soc,
                             direct=True, output=None)
    if gap:
        kvbm = ibzkpts[p1[1]]
        kcbm = ibzkpts[p2[1]]
        vbm = eps[p1]
        cbm = eps[p2]
        subresults = {'vbm_gw': vbm,
                      'cbm_gw': cbm,
                      'dir_gap_gw': gapd,
                      'gap_gw': gap,
                      'kvbm': kvbm,
                      'kcbm': kcbm}
        kd.update({'vbm_gw': 'KVP: GW valence band max. [eV]',
                   'cbm_gw': 'KVP: GW conduction band min. [eV]',
                   'dir_gap_gw': 'KVP: GW direct gap [eV]',
                   'gap_gw': 'KVP: GW gap [eV]',
                   'kvbm': 'k-point of GW valence band max.',
                   'kcbm': 'k-point of GW conduction band min.'})
        results.update(subresults)

    results.update({'efermi_gw_nosoc': efermi_nosoc,
                    'efermi_gw_soc': efermi_soc})
    kd.update({'efermi_gw_nosoc': 'GW Fermi energy w/o soc [eV]',
               'efermi_gw_soc': 'GW Fermi energy [eV]'})
    results['__key_descriptions__'] = kd

    return results


if __name__ == '__main__':
    main.cli()
