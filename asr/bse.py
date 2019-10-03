import numpy as np
from asr.core import command, option, read_json
from click import Choice


@command(module='asr.bse',
         creates=['bse_polx.csv', 'bse_eigx.dat',
                  'bse_poly.csv', 'bse_eigy.dat',
                  'bse_polz.csv', 'bse_eigz.dat'],
         requires=['gs.gpw'],
         resources='480:20h')
@option('--gs', help='Ground state on which BSE is based')
@option('--kptdensity', help='K-point density')
@option('--ecut', help='Plane wave cutoff')
@option('--nv', help='Valence bands included')
@option('--nc', help='Conduction bands included')
@option('--mode', help='Irreducible response',
        type=Choice(['RPA', 'BSE', 'TDHF']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def calculate(gs='gs.gpw', kptdensity=6.0, ecut=50.0, mode='BSE', bandfactor=6,
              nv=4, nc=4):
    """Calculate BSE polarizability"""
    import os
    from ase.io import read
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.bse import BSE
    from gpaw.occupations import FermiDirac
    from pathlib import Path
    import numpy as np
    from asr.core import file_barrier

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()

    ND = np.sum(pbc)
    if ND == 3:
        eta = 0.1
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        truncation = None
    elif ND == 2:
        eta = 0.05

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

        kpts = get_kpts_size(atoms=atoms, kptdensity=20)
        truncation = '2D'
        
    else:
        raise NotImplementedError(
            'asr for BSE not implemented for 0D and 1D structures')

    calc_old = GPAW(gs, txt=None)
    spin = calc_old.get_spin_polarized()
    nval = calc_old.wfs.nvalence
    nocc = int(nval / 2)
    nbands = bandfactor * nocc
    if not Path('gs_bse.gpw').is_file():
        calc = GPAW(
            gs,
            txt='gs_bse.txt',
            fixdensity=True,
            nbands=int(nbands * 1.5),
            convergence={'bands': nbands},
            occupations=FermiDirac(width=1e-4),
            kpts=kpts)
        calc.get_potential_energy()
        with file_barrier('gs_bse.gpw'):
            calc.write('gs_bse.gpw', mode='all')

    if spin:
        f0 = calc.get_occupation_numbers(spin=0)
        f1 = calc.get_occupation_numbers(spin=1)
        n0 = np.where(f0 < 1.0e-6)[0][0]
        n1 = np.where(f1 < 1.0e-6)[0][0]
        valence_bands = [range(n0 - nv, n0), range(n1 - nv, n1)]
        conduction_bands = [range(n0, n0 + nc), range(n1, n1 + nc)]
    else:
        valence_bands = range(nocc - nv, nocc)
        conduction_bands = range(nocc, nocc + nc)

    world.barrier()

    bse = BSE('gs_bse.gpw',
              spinors=True,
              ecut=ecut,
              valence_bands=valence_bands,
              conduction_bands=conduction_bands,
              nbands=nbands,
              mode=mode,
              truncation=truncation,
              txt='bse.txt')

    w_w = np.linspace(-2.0, 8.0, 10001)

    w_w, alphax_w = bse.get_polarizability(eta=eta,
                                           filename='bse_polx.csv',
                                           direction=0,
                                           write_eig='bse_eigx.dat',
                                           pbc=pbc,
                                           w_w=w_w)
    
    w_w, alphay_w = bse.get_polarizability(eta=eta,
                                           filename='bse_poly.csv',
                                           direction=1,
                                           write_eig='bse_eigy.dat',
                                           pbc=pbc,
                                           w_w=w_w)

    w_w, alphaz_w = bse.get_polarizability(eta=eta,
                                           filename='bse_polz.csv',
                                           direction=2,
                                           write_eig='bse_eigz.dat',
                                           pbc=pbc,
                                           w_w=w_w)
    if world.rank == 0:
        os.system('rm gs_bse.gpw')
        os.system('rm gs_nosym.gpw')

                         
def absorption(row, filename, direction='x'):
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.units import alpha, Ha, Bohr
    from ase.io import read

    def ylim(w_w, data, wstart=0.0):
        i = abs(w_w - wstart).argmin()
        x = data[i:]
        x1, x2 = x.real, x.imag
        y = max(x1.max(), x2.max()) * 1.02
        return y

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    dim = np.sum(pbc)

    info = read_json('results-asr.structureinfo.json')
    magstate = info['magstate']

    gsresults = read_json('results-asr.gs.json')
    gap_soc = gsresults['gaps_soc']['gap_dir']
    gap_nosoc = gsresults['gaps_nosoc']['gap_dir']
    qp_gap_soc = gap_soc  # XXX Correct this - QP gap with SOC
    qp_gap_nosoc = gap_soc  # XXX Correct this - QP gap without SOC
    if magstate == 'NM':
        delta_bse = qp_gap_soc - gap_soc
    else:
        delta_bse = qp_gap_nosoc - gap_nosoc
    delta_rpa = qp_gap_nosoc - gap_nosoc

    ax = plt.figure().add_subplot(111)

    data = row.data['results-asr.bse.json'][f'bse_alpha{direction}_w']
    wbse_w = data[:, 0] + delta_bse
    abs_w = 4 * np.pi * data[:, 2]
    if dim == 2:
        abs_w *= wbse_w * alpha / Ha / Bohr * 100
    ybse = ylim(wbse_w, abs_w, 0.0)
    ax.plot(wbse_w, abs_w, '-', c='0.0', label='BSE')

    data = row.data['results-asr.polarizability.json']
    wrpa_w = data['frequencies'] + delta_rpa
    abs_w = 4 * np.pi * data[f'alpha{direction}_w'].imag
    if dim == 2:
        abs_w *= wrpa_w * alpha / Ha / Bohr
    yrpa = ylim(wrpa_w, abs_w, 0.0)
    ax.plot(wrpa_w, abs_w, '-', c='C0', label='RPA')

    ymax = max(ybse, yrpa)

    if magstate == 'NM':
        ax.plot([gap_soc, gap_soc], [0, ymax], '--', c='0.5',
                label='Direct QP gap')
    else:
        ax.plot([gap_nosoc, gap_nosoc], [0, ymax], '--', c='0.5',
                label='Direct QP gap')

    ax.set_xlim(0.0, wbse_w[-1])
    ax.set_ylim(0.0, ymax)
    ax.set_title(f'{direction}-direction')
    ax.set_xlabel('energy [eV]')
    if dim == 2:
        ax.set_ylabel('absorbance [%]')
    else:
        ax.set_ylabel(r'$\varepsilon(\omega)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)

    return ax


def webpanel(row, key_descriptions):
    from functools import partial
    from asr.browser import fig, table
    from ase.io import read

    E_B = table(row, 'Property', ['E_B'], key_descriptions)

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    dim = np.sum(pbc)

    if dim == 2:
        funcx = partial(absorption, direction='x')
        funcz = partial(absorption, direction='z')

        panel = {'title': 'Optical absorption',
                 'columns': [[fig('absx.png'), E_B],
                             [fig('absz.png')]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    else:
        funcx = partial(absorption, direction='x')
        funcy = partial(absorption, direction='y')
        funcz = partial(absorption, direction='z')

        panel = {'title': 'Optical absorption',
                 'columns': [[fig('absx.png'), fig('absz.png')],
                             [fig('absy.png'), E_B]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcy,
                                        'filenames': ['absy.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    return [panel]


@command(module='asr.bse',
         requires=['bse_polx.csv', 'results-asr.gs.json',
                   'results-asr.structureinfo.json'],
         dependencies=['asr.bse@calculate', 'asr.gs', 'asr.structureinfo'],
         webpanel=webpanel)
def main():
    alphax_w = np.loadtxt('bse_polx.csv', delimiter=',')
    data = {'bse_alphax_w': alphax_w.astype(np.float32)}

    from pathlib import Path
    if Path('bse_poly.csv').is_file():
        alphay_w = np.loadtxt('bse_poly.csv', delimiter=',')
        data['bse_alphay_w'] = alphay_w.astype(np.float32)
    if Path('bse_polz.csv').is_file():
        alphaz_w = np.loadtxt('bse_polz.csv', delimiter=',')
        data['bse_alphaz_w'] = alphaz_w.astype(np.float32)
    from asr.core import read_json
                         
    if Path('bse_eigx.dat').is_file():
        E = np.loadtxt('bse_eigx.dat')[0, 1]

        info = read_json('results-asr.structureinfo.json')
        magstate = info['magstate']

        gsresults = read_json('results-asr.gs.json')
        if magstate == 'NM':
            E_B = gsresults['gaps_soc']['gap_dir'] - E
        else:
            E_B = gsresults['gaps_nosoc']['gap_dir'] - E

        data['E_B'] = E_B
        data['__key_descriptions__'] = \
            {'E_B': 'KVP: BSE binding energy (Exc. bind. energy) [eV]'}

    return data


if __name__ == '__main__':
    main.cli()
