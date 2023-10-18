"""Bethe Salpeter absorption spectrum."""
from click import Choice
from asr.core import command, option, file_barrier, ASRResult
from asr.paneldata import BSEResult
from asr.utils.kpts import get_kpts_size


@command(creates=['bse_polx.csv', 'bse_eigx.dat',
                  'bse_poly.csv', 'bse_eigy.dat',
                  'bse_polz.csv', 'bse_eigz.dat'],
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         resources='480:20h')
@option('--gs', help='Ground state on which BSE is based', type=str)
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
@option('--nv_s', help='Valence bands included', type=float)
@option('--nc_s', help='Conduction bands included', type=float)
@option('--mode', help='Irreducible response',
        type=Choice(['RPA', 'BSE', 'TDHF']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def calculate(gs: str = 'gs.gpw', kptdensity: float = 20.0, ecut: float = 50.0,
              mode: str = 'BSE', bandfactor: int = 6,
              nv_s: float = -2.3, nc_s: float = 2.3) -> ASRResult:
    """Calculate BSE polarizability."""
    import os
    from ase.io import read
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.bse import BSE
    from gpaw.occupations import FermiDirac
    from pathlib import Path
    import numpy as np

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    ND = np.sum(pbc)
    if ND == 3:
        eta = 0.1
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        truncation = None
    elif ND == 2:
        eta = 0.05
        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)
        truncation = '2D'

    else:
        raise NotImplementedError(
            'asr for BSE not implemented for 0D and 1D structures')

    calc_gs = GPAW(gs, txt=None)
    spin = calc_gs.get_number_of_spins() == 2
    nval = calc_gs.wfs.nvalence
    nocc = int(nval / 2)
    nbands = bandfactor * nocc
    Nk = len(calc_gs.get_ibz_k_points())
    gap, v, c = bandgap(calc_gs, direct=True, output=None)

    if isinstance(nv_s, float):
        ev = calc_gs.get_eigenvalues(kpt=v[1], spin=v[0])[v[2]]
        nv_sk = np.zeros((spin + 1, Nk), int)
        for s in range(spin + 1):
            for k in range(Nk):
                e_n = calc_gs.get_eigenvalues(kpt=k, spin=s)
                e_n -= ev
                x = e_n[np.where(e_n < 0)]
                x = x[np.where(x > nv_s)]
                nv_sk[s, k] = len(x)
        nv_s = np.max(nv_sk, axis=1)
    if isinstance(nc_s, float):
        ec = calc_gs.get_eigenvalues(kpt=c[1], spin=c[0])[c[2]]
        nc_sk = np.zeros((spin + 1, Nk), int)
        for s in range(spin + 1):
            for k in range(Nk):
                e_n = calc_gs.get_eigenvalues(kpt=k, spin=s)
                e_n -= ec
                x = e_n[np.where(e_n > 0)]
                x = x[np.where(x < nc_s)]
                nc_sk[s, k] = len(x)
        nc_s = np.max(nc_sk, axis=1)

    nv_s = [np.max(nv_s), np.max(nv_s)]
    nc_s = [np.max(nc_s), np.max(nc_s)]

    valence_bands = []
    conduction_bands = []
    for s in range(spin + 1):
        gap, v, c = bandgap(calc_gs, direct=True, spin=s, output=None)
        valence_bands.append(range(c[2] - nv_s[s], c[2]))
        conduction_bands.append(range(c[2], c[2] + nc_s[s]))

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
        with file_barrier(['gs_bse.gpw']):
            calc.write('gs_bse.gpw', mode='all')

    # if spin:
    #     f0 = calc.get_occupation_numbers(spin=0)
    #     f1 = calc.get_occupation_numbers(spin=1)
    #     n0 = np.where(f0 < 1.0e-6)[0][0]
    #     n1 = np.where(f1 < 1.0e-6)[0][0]
    #     valence_bands = [range(n0 - nv, n0), range(n1 - nv, n1)]
    #     conduction_bands = [range(n0, n0 + nc), range(n1, n1 + nc)]
    # else:
    #     valence_bands = range(nocc - nv, nocc)
    #     conduction_bands = range(nocc, nocc + nc)

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


@command(module='asr.bse',
         requires=['bse_polx.csv', 'results-asr.gs.json'],
         dependencies=['asr.bse@calculate', 'asr.gs'],
         returns=BSEResult)
def main() -> BSEResult:
    import numpy as np
    from pathlib import Path
    from asr.core import read_json

    alphax_w = np.loadtxt('bse_polx.csv', delimiter=',')
    data = {'bse_alphax_w': alphax_w.astype(np.float32)}

    if Path('bse_poly.csv').is_file():
        alphay_w = np.loadtxt('bse_poly.csv', delimiter=',')
        data['bse_alphay_w'] = alphay_w.astype(np.float32)
    if Path('bse_polz.csv').is_file():
        alphaz_w = np.loadtxt('bse_polz.csv', delimiter=',')
        data['bse_alphaz_w'] = alphaz_w.astype(np.float32)

    if Path('bse_eigx.dat').is_file():
        E = np.loadtxt('bse_eigx.dat')[0, 1]

        magstateresults = read_json('results-asr.magstate.json')
        magstate = magstateresults['magstate']

        gsresults = read_json('results-asr.gs.json')
        if magstate == 'NM':
            E_B = gsresults['gap_dir'] - E
        else:
            E_B = gsresults['gap_dir_nosoc'] - E

        data['E_B'] = E_B

    return data


if __name__ == '__main__':
    main.cli()
