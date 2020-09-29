from asr.core import command, option, read_json, ASRResult


@command(module='asr.hse',
         dependencies=['asr.structureinfo', 'asr.gs@calculate', 'asr.gs'],
         creates=['hse_nowfs.gpw', 'hse-snapshot.json'],
         requires=['gs.gpw', 'results-asr.gs.json'],
         resources='24:10h')
@option('--kptdensity', help='K-point density', type=float)
@option('--emptybands', help='number of empty bands to include', type=int)
def calculate(kptdensity: float = 8.0, emptybands: int = 20):
    """Calculate HSE corrections."""
    import gpaw.mpi as mpi

    eigs = hse(kptdensity=kptdensity, emptybands=emptybands)
    mpi.world.barrier()
    eigs_soc = hse_spinorbit(eigs)
    mpi.world.barrier()
    results = {'hse_eigenvalues': eigs,
               'hse_eigenvalues_soc': eigs_soc}
    return results


# XXX move to utils? [also in asr.polarizability]
def get_kpts_size(atoms, kptdensity):
    """Try to get a reasonable monkhorst size which hits high symmetry points."""
    from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
    size, offset = k2so(atoms=atoms, density=kptdensity)
    size[2] = 1
    for i in range(2):
        if size[i] % 6 != 0:
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts


def hse(kptdensity, emptybands):
    import numpy as np
    import gpaw.mpi as mpi
    from gpaw import GPAW
    from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues

    convbands = int(emptybands / 2)
    calc = GPAW('gs.gpw', txt=None, parallel={'band': 1, 'kpt': 1})
    atoms = calc.get_atoms()
    pbc = atoms.pbc.tolist()
    ND = np.sum(pbc)
    if ND == 3 or ND == 1:
        kpts = {'density': kptdensity, 'gamma': True, 'even': False}
    elif ND == 2:
        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)

    calc.set(nbands=-emptybands,
             fixdensity=True,
             kpts=kpts,
             convergence={'bands': -convbands},
             txt='hse.txt')
    calc.get_potential_energy()
    calc.write('hse_nowfs.gpw')
    nb = calc.get_number_of_bands()
    result = non_self_consistent_eigenvalues(calc,
                                             'HSE06',
                                             n1=0,
                                             n2=nb - convbands,
                                             snapshot='hse-snapshot.json')
    e_pbe_skn, vxc_pbe_skn, vxc_hse_skn = result
    e_hse_skn = e_pbe_skn - vxc_pbe_skn + vxc_hse_skn

    dct = {}
    if mpi.world.rank == 0:
        dct = dict(vxc_hse_skn=vxc_hse_skn,
                   e_pbe_skn=e_pbe_skn,
                   vxc_pbe_skn=vxc_pbe_skn,
                   e_hse_skn=e_hse_skn)
    return dct


def hse_spinorbit(dct):
    import os
    import numpy as np
    import gpaw.mpi as mpi
    from gpaw import GPAW
    from gpaw.spinorbit import get_spinorbit_eigenvalues as get_soc_eigs
    from asr.magnetic_anisotropy import get_spin_axis, get_spin_index

    if not os.path.isfile('hse_nowfs.gpw'):
        return

    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    if mpi.world.rank in ranks:
        calc = GPAW('hse_nowfs.gpw', communicator=comm, txt=None)
        e_skn = dct.get('e_hse_skn')
        dct_soc = {}
        theta, phi = get_spin_axis()

        e_mk, s_kvm = get_soc_eigs(calc, gw_kn=e_skn, return_spin=True,
                                   bands=np.arange(e_skn.shape[2]),
                                   theta=theta, phi=phi)
        dct_soc['e_hse_mk'] = e_mk
        dct_soc['s_hse_mk'] = s_kvm[:, get_spin_index(), :].transpose()

        return dct_soc


def MP_interpolate(calc, delta_skn, lb, ub):
    """Interpolate corrections to band patch.

    Calculates band stucture along the same band path used for PBE
    by interpolating a correction onto the PBE band structure.
    """
    import numpy as np
    import gpaw.mpi as mpi
    from gpaw import GPAW
    from gpaw.spinorbit import get_spinorbit_eigenvalues as get_soc_eigs
    from ase.dft.kpoints import (get_monkhorst_pack_size_and_offset,
                                 monkhorst_pack_interpolate)
    from asr.core import singleprec_dict
    from asr.magnetic_anisotropy import get_spin_axis

    bandrange = np.arange(lb, ub)
    # read PBE (without SOC)
    results_bandstructure = read_json('results-asr.bandstructure.json')
    path = results_bandstructure['bs_nosoc']['path']
    e_pbe_skn = results_bandstructure['bs_nosoc']['energies']

    size, offset = get_monkhorst_pack_size_and_offset(calc.get_bz_k_points())
    bz2ibz = calc.get_bz_to_ibz_map()
    icell = calc.atoms.get_reciprocal_cell()
    eps = monkhorst_pack_interpolate(path.kpts, delta_skn.transpose(1, 0, 2),
                                     icell, bz2ibz, size, offset)
    delta_interp_skn = eps.transpose(1, 0, 2)
    e_int_skn = e_pbe_skn[:, :, bandrange] + delta_interp_skn
    dct = dict(e_int_skn=e_int_skn, path=path)

    # add SOC from bs.gpw
    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    if mpi.world.rank in ranks:
        calc = GPAW('bs.gpw', communicator=comm, txt=None)
        theta, phi = get_spin_axis()
        e_int_mk, s_int_mk = get_soc_eigs(calc, gw_kn=e_int_skn,
                                          return_spin=True,
                                          bands=bandrange,
                                          theta=theta, phi=phi)
        dct.update(e_int_mk=e_int_mk, s_int_mk=s_int_mk)

    results = {}
    results['bandstructure'] = singleprec_dict(dct)

    return results


def bs_hse(row,
           filename='hse-bs.png',
           figsize=(5.5, 5),
           fontsize=10,
           show_legend=True,
           s=0.5):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    data = row.data.get('results-asr.hse.json')
    path = data['bandstructure']['path']
    mpl.rcParams['font.size'] = fontsize
    ef = data['efermi_hse_soc']

    reference = row.get('evac', row.get('efermi'))
    if row.get('evac') is not None:
        label = r'$E - E_\mathrm{vac}$ [eV]'
    else:
        label = r'$E - E_\mathrm{F}$ [eV]'

    emin = row.get('vbm_hse', ef) - 3 - reference
    emax = row.get('cbm_hse', ef) + 3 - reference
    e_mk = data['bandstructure']['e_int_mk'] - reference
    x, X, labels = path.get_linear_kpoint_axis()

    # hse with soc
    hse_style = dict(
        color='C1',
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    for e_m in e_mk:
        ax.plot(x, e_m, **hse_style)
    ax.set_ylim([emin, emax])
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylabel(label)
    ax.set_xlabel('$k$-points')
    ax.set_xticks(X)
    ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in labels])

    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    ax.axhline(ef - reference, c='C1', ls=':')
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
        ax = add_bs_pbe(row, ax, reference=row.get('evac', row.get('efermi')),
                        color=[0.8, 0.8, 0.8])

    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    ax.plot([], [], **hse_style, label='HSE')
    plt.legend(loc='upper right')

    if not show_legend:
        ax.legend_.remove()
    plt.savefig(filename, bbox_inches='tight')


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig, table

    if row.get('gap_hse', 0) > 0.0:
        hse = table(row, 'Property',
                    ['gap_hse', 'gap_dir_hse'],
                    kd=key_descriptions)

        if row.get('evac'):
            hse['rows'].extend(
                [['Valence band maximum wrt. vacuum level (HSE)',
                  f'{row.vbm_hse - row.evac:.2f} eV'],
                 ['Conduction band minimum wrt. vacuum level (HSE)',
                  f'{row.cbm_hse - row.evac:.2f} eV']])
        else:
            hse['rows'].extend(
                [['Valence band maximum wrt. Fermi level (HSE)',
                  f'{row.vbm_hse - row.efermi:.2f} eV'],
                 ['Conduction band minimum wrt. Fermi level (HSE)',
                  f'{row.cbm_hse - row.efermi:.2f} eV']])
    else:
        hse = table(row, 'Property',
                    [],
                    kd=key_descriptions)

    panel = {'title': 'Electronic band structure (HSE)',
             'columns': [[fig('hse-bs.png')],
                         [fig('bz-with-gaps.png'), hse]],
             'plot_descriptions': [{'function': bs_hse,
                                    'filenames': ['hse-bs.png']}],
             'sort': 15}

    if row.get('gap_hse'):
        rows = [['Band gap (HSE)', f'{row.gap_hse:0.2f} eV']]
        summary = {'title': 'Summary',
                   'columns': [[{'type': 'table',
                                 'header': ['Electronic properties', ''],
                                 'rows': rows}]],
                   'sort': 11}
        return [panel, summary]

    return [panel]


class Result(ASRResult):

    formats = {"ase_webpanel": webpanel}


@command(module='asr.hse',
         dependencies=['asr.hse@calculate', 'asr.bandstructure'],
         requires=['bs.gpw',
                   'hse_nowfs.gpw',
                   'results-asr.bandstructure.json',
                   'results-asr.hse@calculate.json'],
         resources='1:10m',
         returns=Result)
def main():
    """Interpolate HSE band structure along a given path."""
    import numpy as np
    from gpaw import GPAW
    from asr.utils import fermi_level
    from ase.dft.bandgap import bandgap

    # interpolate band structure
    calc = GPAW('hse_nowfs.gpw', txt=None)
    results_hse = read_json('results-asr.hse@calculate.json')
    data = results_hse['hse_eigenvalues']
    nbands = data['e_hse_skn'].shape[2]
    delta_skn = data['vxc_hse_skn'] - data['vxc_pbe_skn']
    results = MP_interpolate(calc, delta_skn, 0, nbands)

    # get gap, cbm, vbm, etc...
    results['__key_descriptions__'] = {}
    results_calc = read_json('results-asr.hse@calculate.json')
    eps_skn = results_calc['hse_eigenvalues']['e_hse_skn']
    calc = GPAW('hse_nowfs.gpw', txt=None)
    ibzkpts = calc.get_ibz_k_points()
    efermi_nosoc = fermi_level(calc, eigenvalues=eps_skn,
                               nspins=eps_skn.shape[0])
    gap, p1, p2 = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                             direct=True, output=None)
    if gap:
        kvbm_nosoc = ibzkpts[p1[1]]  # k coordinates of vbm
        kcbm_nosoc = ibzkpts[p2[1]]  # k coordinates of cbm
        vbm = eps_skn[p1]
        cbm = eps_skn[p2]
        subresults = {'vbm_hse_nosoc': vbm,
                      'cbm_hse_nosoc': cbm,
                      'gap_dir_hse_nosoc': gapd,
                      'gap_hse_nosoc': gap,
                      'kvbm_nosoc': kvbm_nosoc,
                      'kcbm_nosoc': kcbm_nosoc}
        kd = {'vbm_hse_nosoc': 'HSE valence band max. w/o soc [eV]',
              'cbm_hse_nosoc': 'HSE condution band min. w/o soc [eV]',
              'gap_dir_hse_nosoc': 'HSE direct gap w/o soc [eV]',
              'gap_hse_nosoc': 'HSE gap w/o soc [eV]',
              'kvbm_nosoc': 'k-point of HSE valence band max. w/o soc',
              'kcbm_nosoc': 'k-point of HSE conduction band min. w/o soc'}
        results.update(subresults)
        results['__key_descriptions__'].update(kd)

    eps = results_calc['hse_eigenvalues_soc']['e_hse_mk']
    eps = eps.transpose()[np.newaxis]  # e_skm, dummy spin index
    efermi_soc = fermi_level(calc, eigenvalues=eps, nspins=2)
    gap, p1, p2 = bandgap(eigenvalues=eps, efermi=efermi_soc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps, efermi=efermi_soc,
                             direct=True, output=None)
    if gap:
        kvbm = ibzkpts[p1[1]]
        kcbm = ibzkpts[p2[1]]
        vbm = eps[p1]
        cbm = eps[p2]
        subresults = {'vbm_hse': vbm,
                      'cbm_hse': cbm,
                      'gap_dir_hse': gapd,
                      'gap_hse': gap,
                      'kvbm': kvbm,
                      'kcbm': kcbm}
        kd = {'vbm_hse': 'KVP: HSE valence band max. [eV]',
              'cbm_hse': 'KVP: HSE conduction band min. [eV]',
              'gap_dir_hse': 'KVP: HSE direct gap [eV]',
              'gap_hse': 'KVP: HSE gap [eV]',
              'kvbm': 'k-point of HSE valence band max.',
              'kcbm': 'k-point of HSE conduction band min.'}
        results.update(subresults)
        results['__key_descriptions__'].update(kd)

    subresults = {'efermi_hse_nosoc': efermi_nosoc,
                  'efermi_hse_soc': efermi_soc}
    kd = {'efermi_hse_nosoc': 'HSE Fermi energy w/o soc [eV]',
          'efermi_hse_soc': 'HSE Fermi energy [eV]'}
    results.update(subresults)
    results['__key_descriptions__'].update(kd)

    return results


if __name__ == '__main__':
    main.cli()
