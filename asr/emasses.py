"""Effective masses."""
from asr.core import command, option, DictStr
from asr.result.resultdata import (EmassesResult, ValidateResult, model)
from asr.extra_fluff import MAXMASS, evalmodel


class NoGapError(Exception):
    pass


def set_default(settings):
    if 'erange1' not in settings:
        settings['erange1'] = 250e-3
    if 'nkpts1' not in settings:
        settings['nkpts1'] = 19
    if 'erange2' not in settings:
        settings['erange2'] = 1e-3
    if 'nkpts2' not in settings:
        settings['nkpts2'] = 9


@command(module='asr.emasses',
         requires=['gs.gpw', 'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.structureinfo',
                       'asr.magnetic_anisotropy',
                       'asr.gs'],
         creates=['em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw'])
@option('--gpwfilename', type=str,
        help='GS Filename')
@option('-s', '--settings', help='Settings for the two refinements',
        type=DictStr())
def refine(gpwfilename: str = 'gs.gpw',
           settings: dict = {
               'erange1': 250e-3,
               'nkpts1': 19,
               'erange2': 1e-3,
               'nkpts2': 9}) -> EmassesResult:
    """Take a bandstructure and calculate more kpts around the vbm and cbm."""
    from asr.utils.gpw2eigs import calc2eigs
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    from gpaw import GPAW
    import os.path
    set_default(settings)
    socs = [True]

    for soc in socs:
        theta, phi = get_spin_axis()
        calc = GPAW(gpwfilename, txt=None)
        eigenvalues, efermi = calc2eigs(calc=calc, soc=soc,
                                        theta=theta, phi=phi)
        gap, _, _ = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                            output=None)
        if not gap > 0:
            raise NoGapError('Gap was zero: {}'.format(gap))

        for bt in ['vb', 'cb']:
            name = get_name(soc=soc, bt=bt)
            gpw2 = name + '.gpw'

            if os.path.exists(gpw2):
                continue
            gpwrefined = preliminary_refine(gpw=gpwfilename, soc=soc,
                                            bandtype=bt, settings=settings)
            nonsc_sphere(gpw=gpwrefined, fallback=gpwfilename, soc=soc,
                         bandtype=bt, settings=settings)


def get_name(soc, bt):
    return 'em_circle_{}_{}'.format(bt, ['nosoc', 'soc'][soc])


def preliminary_refine(gpw='gs.gpw', soc=True, bandtype=None, settings=None):
    from gpaw import GPAW
    import numpy as np
    from asr.utils.gpw2eigs import calc2eigs
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    # Get calc and current kpts
    calc = GPAW(gpw, txt=None)
    ndim = calc.atoms.pbc.sum()

    k_kc = calc.get_bz_k_points()
    cell_cv = calc.atoms.get_cell()

    # Find energies and VBM/CBM
    theta, phi = get_spin_axis()
    e_skn, efermi = calc2eigs(calc, soc=soc, theta=theta, phi=phi)
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]
    gap, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn, efermi=efermi,
                                              output=None)
    # Make a sphere of kpts of high density
    erange = settings['erange1']
    nkpts = settings['nkpts1']
    nkpts = nkpts + (1 - (nkpts % 2))
    # Ensure that we include the found VBM/CBM
    assert nkpts % 2 != 0
    ksphere = kptsinsphere(cell_cv, npoints=nkpts,
                           erange=erange, m=1.0,
                           dimensionality=ndim)

    # Position sphere around VBM if bandtype == vb
    # Else around CBM
    if bandtype == 'vb':
        newk_kc = k_kc[k1] + ksphere
    elif bandtype == 'cb':
        newk_kc = k_kc[k2] + ksphere
    else:
        raise ValueError(f'Bandtype "{bandtype}" not recognized')

    # Calculate energies for new k-grid
    fname = '_refined'
    calc.set(kpts=newk_kc,
             symmetry='off',
             txt=fname + '.txt',
             fixdensity=True)
    atoms = calc.get_atoms()
    atoms.get_potential_energy()
    calc.write(fname + '.gpw')
    return fname + '.gpw'


def get_gapskn(calc, fallback=None, soc=True):
    import numpy as np
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    from asr.utils.gpw2eigs import calc2eigs
    from ase.parallel import parprint

    theta, phi = get_spin_axis()
    e_skn, efermi = calc2eigs(calc, soc=soc, theta=theta, phi=phi)
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis, :, :]

    gap, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn, efermi=efermi,
                                              output=None)

    if np.allclose(gap, 0) and fallback is not None:
        parprint("Something went wrong. Using fallback gpw.")
        theta, phi = get_spin_axis()
        e_skn, efermi = calc2eigs(fallback, soc=soc, theta=theta, phi=phi)
        if e_skn.ndim == 2:
            e_skn = e_skn[np.newaxis, :, :]

        gap, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn, efermi=efermi,
                                                  output=None)
    if np.allclose(gap, 0):
        raise ValueError(f"Gap is still zero!")

    return gap, (s1, k1, n1), (s2, k2, n2)


def nonsc_sphere(gpw='gs.gpw', fallback='gs.gpw', soc=False,
                 bandtype=None, settings=None):
    """Non sc calculation for kpts in a sphere around the VBM/CBM.

    Writes the files:

    * em_circle_vb_soc.gpw
    * em_circle_cb_soc.gpw
    * em_circle_vb_nosoc.gpw
    * em_circle_cb_nosoc.gpw

    Parameters
    ----------
    gpw: str
        gpw filename
    soc: bool
        spinorbit coupling
    bandtype: None or 'cb' or 'vb'
        Which bandtype do we do calculations for, if None is done for
        for both cb and vb
    """
    from gpaw import GPAW
    calc = GPAW(gpw, txt=None)
    ndim = calc.atoms.pbc.sum()

    # Check that 1D: Only z-axis, 2D: Only x- and y-axis
    if ndim == 1:
        pbc = calc.atoms.pbc
        assert not pbc[0] and not pbc[1] and pbc[2]
    elif ndim == 2:
        pbc = calc.atoms.pbc
        assert pbc[0] and pbc[1] and not pbc[2]

    k_kc = calc.get_bz_k_points()
    cell_cv = calc.atoms.get_cell()

    nkpts = settings['nkpts2']
    erange = settings['erange2']
    kcirc_kc = kptsinsphere(cell_cv, dimensionality=ndim,
                            erange=erange, npoints=nkpts)

    gap, (s1, k1, n1), (s2, k2, n2) = get_gapskn(calc, fallback=None, soc=soc)

    k1_c = k_kc[k1]
    k2_c = k_kc[k2]

    bandtypes, ks = get_bt_ks(bandtype, k1_c, k2_c)

    for bt, k_c in zip(bandtypes, ks):
        name = get_name(soc=soc, bt=bt)
        calc.set(kpts=kcirc_kc + k_c,
                 symmetry='off',
                 txt=name + '.txt',
                 fixdensity=True)
        atoms = calc.get_atoms()
        atoms.get_potential_energy()
        calc.write(name + '.gpw')


def kptsinsphere(cell_cv, npoints=9, erange=1e-3, m=1.0, dimensionality=3):
    import numpy as np
    from ase.units import Hartree, Bohr
    from ase.dft.kpoints import kpoint_convert

    a = np.linspace(-1, 1, npoints)
    X, Y, Z = np.meshgrid(a, a, a)

    na = np.logical_and
    if dimensionality == 2:
        indices = na(X**2 + Y**2 <= 1.0, Z == 0)
    elif dimensionality == 1:
        indices = na(Z**2 <= 1.0, na(X == 0, Y == 0))
    else:
        indices = X**2 + Y**2 + Z**2 <= 1.0

    x, y, z = X[indices], Y[indices], Z[indices]
    kpts_kv = np.vstack([x, y, z]).T
    kr = np.sqrt(2 * m * erange / Hartree)
    kpts_kv *= kr
    kpts_kv /= Bohr
    kpts_kc = kpoint_convert(cell_cv=cell_cv, ckpts_kv=kpts_kv)
    return kpts_kc


def get_bt_ks(bandtype, k1_c, k2_c):
    if bandtype is None:
        bandtypes = ('vb', 'cb')
        ks = (k1_c, k2_c)
    elif bandtype == 'vb':
        bandtypes = ('vb',)
        ks = (k1_c, )
    elif bandtype == 'cb':
        bandtypes = ('cb', )
        ks = (k2_c, )
    return bandtypes, ks


def maeformat(mae):
    import numpy as np
    f10 = np.log(mae) / np.log(10)
    f10 = round(f10)
    mae = mae / 10**(f10)
    if mae < 1:
        f10 -= 1
        mae *= 10

    maestr = round(mae, 2)

    maestr = str(maestr) + f'e{f10}'
    return maestr


@command('asr.emasses',
         requires=['em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw',
                   'gs.gpw', 'results-asr.structureinfo.json',
                   'results-asr.gs.json',
                   'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.emasses@refine',
                       'asr.gs@calculate',
                       'asr.gs',
                       'asr.structureinfo',
                       'asr.magnetic_anisotropy'])
@option('--gpwfilename', type=str,
        help='GS Filename')
def main(gpwfilename: str = 'gs.gpw') -> EmassesResult:
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    import traceback
    socs = [True]

    good_results = {}
    for soc in socs:
        theta, phi = get_spin_axis()
        eigenvalues, efermi = gpw2eigs(gpw=gpwfilename, soc=soc,
                                       theta=theta, phi=phi)
        gap, _, _ = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                            output=None)
        if not gap > 0:
            raise NoGapError('Gap was zero')
        for bt in ['vb', 'cb']:
            name = get_name(soc=soc, bt=bt)
            gpw2 = name + '.gpw'
            try:
                masses = embands(gpw2,
                                 soc=soc,
                                 bandtype=bt)

                # This function modifies the last argument
                unpack_masses(masses, soc, bt, good_results)
            except ValueError:
                tb = traceback.format_exc()
                print(gpw2 + ':\n' + '=' * len(gpw2) + '\n', tb)

    return good_results


def unpack_masses(masses, soc, bt, results_dict):
    # Structure of 'masses' object:
    # There is an 'indices' key which tells you at which spin-k indices
    # effective masses have been calculated
    # At a given index there is saved the dict returned by the em function:
    # out = dict(mass_u=masses, eigenvectors_vu=vecs,
    #            ke_v=kmax,
    #            c=c,
    #            r=r)
    # We want to flatten this structure so that results_dict contains
    # the masses directly
    import numpy as np

    for ind in masses['indices']:
        out_dict = masses[ind]
        index = str(ind)
        socpre = 'soc' if soc else 'nosoc'
        prefix = bt + '_' + socpre + '_'
        offset = out_dict['offset']

        results_dict[index] = {}

        mass_u = list(out_dict['mass_u'])

        for u, m in enumerate(mass_u):
            if np.isnan(m):
                mass_u[u] = None

        results_dict[index][prefix + 'effmass_dir1'] = mass_u[0]
        results_dict[index][prefix + 'effmass_dir2'] = mass_u[1]
        results_dict[index][prefix + 'effmass_dir3'] = mass_u[2]

        if offset == 0:
            results_dict[f'emass_{bt}_dir1'] = mass_u[0]
            results_dict[f'emass_{bt}_dir2'] = mass_u[1]
            results_dict[f'emass_{bt}_dir3'] = mass_u[2]

        vecs = out_dict['eigenvectors_vu']
        results_dict[index][prefix + 'eigenvectors_vdir1'] = vecs[:, 0]
        results_dict[index][prefix + 'eigenvectors_vdir2'] = vecs[:, 1]
        results_dict[index][prefix + 'eigenvectors_vdir3'] = vecs[:, 2]
        results_dict[index][prefix + 'spin'] = ind[0]
        results_dict[index][prefix + 'bandindex'] = ind[1]
        results_dict[index][prefix + 'kpt_v'] = out_dict['ke_v']
        results_dict[index][prefix + 'fitcoeff'] = out_dict['c']
        results_dict[index][prefix + '2ndOrderFit'] = out_dict['c2']
        results_dict[index][prefix + 'mass_u'] = mass_u
        results_dict[index][prefix + 'bzcuts'] = out_dict['bs_along_emasses']
        results_dict[index][prefix + 'fitkpts_kv'] = out_dict['fitkpts_kv']
        results_dict[index][prefix + 'fite_k'] = out_dict['fite_k']
        results_dict[index][prefix + '2ndOrderr2'] = out_dict['r2']
        results_dict[index][prefix + '3rdOrderr2'] = out_dict['r']
        results_dict[index][prefix + '2ndOrderMAE'] = out_dict['mae2']
        results_dict[index][prefix + '3rdOrderMAE'] = out_dict['mae3']
        results_dict[index][prefix + 'wideareaMAE'] = out_dict['wideareaMAE']


def embands(gpw, soc, bandtype, delta=0.1):
    """Effective masses for bands within delta of extrema.

    Parameters
    ----------
        gpw: str
            name of gpw filename
        soc: bool
            include spin-orbit coupling
        bandtype: 'vb' or 'cb'
            type of band
        efermi: float, optional
            fermi level (takes it from gpw if None)
        delta: float, optional
            bands within this value (in eV) is included in the em fit
            default is 0.1 eV

    """
    from gpaw import GPAW
    from asr.utils.gpw2eigs import gpw2eigs
    import numpy as np
    from ase.dft.kpoints import kpoint_convert
    from ase.units import Bohr, Hartree
    from asr.magnetic_anisotropy import get_spin_axis
    calc = GPAW(gpw, txt=None)
    ndim = calc.atoms.pbc.sum()

    theta, phi = get_spin_axis()
    e_skn, efermi = gpw2eigs(gpw, soc=soc, theta=theta, phi=phi)
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]

    vb_ind, cb_ind = get_vb_cb_indices(e_skn=e_skn, efermi=efermi, delta=delta)

    indices = vb_ind if bandtype == 'vb' else cb_ind
    atoms = calc.get_atoms()
    cell_cv = atoms.get_cell()
    bz_kc = calc.get_bz_k_points()

    bz_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=bz_kc)
    masses = {'indices': indices}
    for offset, b in enumerate(indices):
        e_k = e_skn[b[0], :, b[1]]
        masses[b] = em(kpts_kv=bz_kv * Bohr,
                       eps_k=e_k / Hartree, bandtype=bandtype, ndim=ndim)

        calc_bs = calculate_bs_along_emass_vecs
        if offset == 0:
            nbands = len(indices)
        else:
            nbands = 1
        masses[b]['bs_along_emasses'] = calc_bs(masses[b],
                                                soc, bandtype, calc,
                                                spin=b[0],
                                                band=b[1],
                                                nbands=nbands)
        masses[b]['wideareaMAE'] = wideMAE(masses[b], bandtype,
                                           cell_cv)
        masses[b]['offset'] = offset

    return masses


def calculate_bs_along_emass_vecs(masses_dict, soc,
                                  bt, calc,
                                  spin, band,
                                  erange=250e-3, npoints=91,
                                  nbands=1):
    from pathlib import Path
    from ase.units import Hartree, Bohr
    from ase.dft.kpoints import kpoint_convert
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis, get_spin_index
    from asr.core import file_barrier
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    import numpy as np
    cell_cv = calc.get_atoms().get_cell()

    results_dicts = []
    for u, mass in enumerate(masses_dict['mass_u']):
        if mass is np.nan or np.isnan(mass) or mass is None:
            continue
        identity = f'em_bs_spin={spin}_band={band}_bt={bt}_dir={u}_soc={soc}'
        name = f'{identity}.gpw'

        if not Path(name).is_file():
            with file_barrier([name]):
                kmax = np.sqrt(2 * abs(mass) * erange / Hartree)
                _max = np.sqrt(2 * MAXMASS * erange / Hartree)
                if kmax > _max:
                    kmax = _max
                assert not np.isnan(kmax)
                kd_v = masses_dict['eigenvectors_vu'][:, u]
                assert not (np.isnan(kd_v)).any()
                k_kv = (np.linspace(-1, 1, npoints) * kmax * kd_v.reshape(3, 1)).T
                k_kv += masses_dict['ke_v']
                k_kv /= Bohr
                assert not (np.isnan(k_kv)).any()
                k_kc = kpoint_convert(cell_cv=cell_cv, ckpts_kv=k_kv)
                assert not (np.isnan(k_kc)).any()
                atoms = calc.get_atoms()
                for i, pb in enumerate(atoms.pbc):
                    if not pb:
                        k_kc[:, i] = 0
                assert not (np.isnan(k_kc)).any()
                calc.set(kpts=k_kc, symmetry='off',
                         txt=f'{identity}.txt', fixdensity=True)
                atoms.get_potential_energy()
                calc.write(name)

        calc_serial = GPAW(name, txt=None, communicator=serial_comm)
        k_kc = calc_serial.get_bz_k_points()
        theta, phi = get_spin_axis()
        e_km, _, s_kvm = calc2eigs(calc_serial, soc=soc, return_spin=True,
                                   theta=theta, phi=phi)

        sz_km = s_kvm[:, get_spin_index(), :]

        dct = dict(bt=bt,
                   kpts_kc=k_kc,
                   e_k=e_km[:, band],
                   spin_k=sz_km[:, band])
        if bt == "vb":
            dct['e_km'] = e_km[:, band - nbands + 1:band + 1]
            dct['spin_km'] = sz_km[:, band - nbands + 1:band + 1]
        else:
            dct['e_km'] = e_km[:, band:band + nbands]
            dct['spin_km'] = sz_km[:, band:band + nbands]

        results_dicts.append(dct)

    return results_dicts


def get_vb_cb_indices(e_skn, efermi, delta):
    """Find CB and VB within a distance of delta of the CB and VB extrema.

    Parameters
    ----------
        e_skn: (ns, nk, nb)-shape ndarray
            eigenvalues
        efermi: float
            fermi level
        delta: float
            bands within delta of the extrema are included
    Returns
    -------
        vb_indices, cb_indices: [(spin, band), ..], [(spin, band), ...]
            spin and band indices (aka as SBandex) for VB and CB, respectively

    """
    import numpy as np
    from ase.dft.bandgap import bandgap
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]
    gap, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn,
                                              efermi=efermi, output=None)

    if not gap > 0:
        raise ValueError('Band gap is zero')

    cbm = e_skn[s2, k2, n2]
    vbm = e_skn[s1, k1, n1]

    cb_sn = e_skn[:, k2, n2:]
    vb_sn = e_skn[:, k1, :n1 + 1]
    cbs, cbn = np.where(cb_sn <= cbm + delta)
    cbn += n2
    cb_indices = list(zip(cbs, cbn))

    vbs, vbn = np.where(vb_sn >= vbm - delta)
    vb_indices = list(reversed(list(zip(vbs, vbn))))
    return vb_indices, cb_indices


def em(kpts_kv, eps_k, bandtype=None, ndim=3):
    """Fit 2nd and 3rd order polynomial to eps_k.

    Parameters
    ----------
        kpts_kv: (nk, 3)-shape ndarray
            k-points in cartesian coordinates (in units of 1 / Bohr)
        eps_k: (nk,)-shape ndarray
            eigenvalues (in units of Hartree)

    Returns
    -------
        out: dct
            - effective masses in units of m_e
            - eigenvectors in cartesian coordinates
            - k-pot extremum in cartesian coordinates (units of 1 / Bohr)

    """
    import numpy as np
    from ase.parallel import parprint
    c, r, rank, s, = fit(kpts_kv, eps_k, thirdorder=False)
    assert (r < 1e-3).all()
    dxx = 2 * c[0]
    dyy = 2 * c[1]
    dzz = 2 * c[2]
    dxy = c[3]
    dxz = c[4]
    dyz = c[5]

    mae2 = np.mean(np.abs(eps_k - evalmodel(kpts_kv, c, thirdorder=False)))

    xm, ym, zm = get_2nd_order_extremum(c, ndim=ndim)
    ke2_v = np.array([xm, ym, zm])

    c3, r3, rank3, s3 = fit(kpts_kv, eps_k, thirdorder=True)

    mae3 = np.mean(np.abs(eps_k - evalmodel(kpts_kv, c3, thirdorder=True)))

    f3xx, f3yy, f3zz, f3xy = c3[:4]
    f3xz, f3yz, f3x, f3y = c3[4:8]
    f3z, f30, f3xxx, f3yyy = c3[8:12]
    f3zzz, f3xxy, f3xxz, f3yyx, f3yyz, f3zzx, f3zzy, f3xyz = c3[12:]

    extremum_type = get_extremum_type(dxx, dyy, dzz, dxy, dxz, dyz, ndim=ndim)
    if extremum_type == 'saddlepoint':
        parprint(f'Found a saddlepoint for bandtype {bandtype}')
    xm, ym, zm = get_3rd_order_extremum(xm, ym, zm, c3,
                                        extremum_type, ndim=ndim)
    ke_v = np.array([xm, ym, zm])

    assert not (np.isnan(ke_v)).any()

    d3xx = (2 * f3xx) + (6 * f3xxx * xm) + (2 * f3xxy * ym) + (2 * f3xxz * zm)
    d3yy = (2 * f3yy) + (6 * f3yyy * ym) + (2 * f3yyx * xm) + (2 * f3yyz * zm)
    d3zz = (2 * f3zz) + (6 * f3zzz * zm) + (2 * f3zzx * xm) + (2 * f3zzy * ym)
    d3xy = f3xy + (2 * f3xxy * xm) + (2 * f3yyx * ym) + (f3xyz * zm)
    d3xz = f3xz + (2 * f3xxz * xm) + (2 * f3zzx * zm) + (f3xyz * ym)
    d3yz = f3yz + (2 * f3yyz * ym) + (2 * f3zzy * zm) + (f3xyz * xm)

    hessian3 = np.array([[d3xx, d3xy, d3xz],
                         [d3xy, d3yy, d3yz],
                         [d3xz, d3yz, d3zz]])

    v3_n, w3_vn = np.linalg.eigh(hessian3)
    assert not (np.isnan(w3_vn)).any()

    hessian = np.array([[dxx, dxy, dxz],
                        [dxy, dyy, dyz],
                        [dxz, dyz, dzz]])
    v2_n, vecs = np.linalg.eigh(hessian)
    mass2_u = np.zeros_like(v2_n)
    npis = np.isclose

    v3_n[npis(v3_n, 0)] = np.nan
    v2_n[npis(v2_n, 0)] = np.nan

    mass2_u = 1 / v2_n
    mass_u = 1 / v3_n

    sort_args = np.argsort(mass_u)

    mass_u = mass_u[sort_args]
    w3_vn = w3_vn[:, sort_args]

    out = dict(mass_u=mass_u,
               eigenvectors_vu=w3_vn,
               ke_v=ke_v,
               c=c3,
               r=r3,
               mass2_u=mass2_u,
               eigenvectors2_vu=vecs,
               ke2_v=ke2_v,
               c2=c,
               r2=r,
               fitkpts_kv=kpts_kv,
               fite_k=eps_k,
               mae2=mae2,
               mae3=mae3)

    return out


def get_extremum_type(dxx, dyy, dzz, dxy, dxz, dyz, ndim=3):
    # Input: 2nd order derivatives at the extremum point
    import numpy as np
    if ndim == 3:
        hessian = np.array([[dxx, dxy, dxz],
                            [dxy, dyy, dyz],
                            [dxz, dyz, dzz]])
        vals, vecs = np.linalg.eigh(hessian)
        saddlepoint = not (np.sign(vals[0]) == np.sign(vals[1])
                           and np.sign(vals[0]) == np.sign(vals[2]))

        if saddlepoint:
            etype = 'saddlepoint'
        elif (vals < 0).all():
            etype = 'max'
        elif (vals > 0).all():
            etype = 'min'
        else:
            msg = 'Extremum type could not be' \
                  + 'found for hessian: {}'.format(hessian)
            raise ValueError(msg)
        return etype
    elif ndim == 2:
        # Assume x and y axis are the periodic directions
        hessian = np.array([[dxx, dxy],
                            [dxy, dyy]])
        det = np.linalg.det(hessian)
        if det < 0:
            etype = 'saddlepoint'
        elif dxx < 0 and dyy < 0:
            etype = 'max'
        elif dxx > 0 and dyy > 0:
            etype = 'min'
        else:
            raise ValueError('Extremum type could not'
                             + 'be determined for hessian: {}'.format(hessian))
        return etype
    elif ndim == 1:
        # Assume z axis is the periodic direction
        if dzz < 0:
            etype = 'max'
        else:
            etype = 'min'
        return etype


def get_2nd_order_extremum(c, ndim=3):
    import numpy as np
    # fit is
    # fxx x^2 + fyy y^2 + fzz z^2 +
    # fxy xy + fxz xz + fyz yz + fx x + fy y + fz z + f0
    assert len(c) == 10
    fxx, fyy, fzz, fxy, fxz, fyz, fx, fy, fz, f0 = c

    if ndim == 3:
        ma = np.array([[2 * fxx, fxy, fxz],
                       [fxy, 2 * fyy, fyz],
                       [fxz, fyz, 2 * fzz]])

        v = np.array([-fx, -fy, -fz])

        min_pos = np.linalg.solve(ma, v)

        return min_pos

    elif ndim == 2:
        # Assume x and y are periodic directions
        ma = np.array([[2 * fxx, fxy],
                       [fxy, 2 * fyy]])
        v = np.array([-fx, -fy])
        min_pos = np.linalg.solve(ma, v)

        return np.array([min_pos[0], min_pos[1], 0.0])

    elif ndim == 1:
        # Assume z is periodic direction
        return np.array([0.0, 0.0, -fz / (2 * fzz)])


def get_3rd_order_extremum(xm, ym, zm, c, extremum_type, ndim=3):
    if extremum_type == 'saddlepoint':
        return xm, ym, zm

    import numpy as np
    from scipy import optimize

    assert len(c) == 20

    if ndim == 3:
        def get_v(kpts):
            k = np.asarray(kpts)
            if k.ndim == 1:
                k = k[np.newaxis]
            return model(k)
    elif ndim == 2:
        # Assume x and y are periodic
        # Remove z-dependence
        def get_v(kpts):
            k = np.asarray(kpts)
            if k.ndim == 1:
                k = k[np.newaxis]
            m = model(k)
            return m
    elif ndim == 1:
        # Assume z is periodic
        # Remove x, y - dependence
        def get_v(kpts):
            k = np.asarray(kpts)
            if k.ndim == 1:
                k = k[np.newaxis]
            m = model(k)
            return m

    # We want to use a minimization function from scipy
    # so if the extremum type is a 'max' we need to multiply
    # the function by - 1
    if extremum_type == 'max':
        def func(v):
            return -1 * np.dot(get_v(v), c)
    else:
        def func(v):
            return np.dot(get_v(v), c)

    x0 = np.array([xm, ym, zm])
    x, y, z = optimize.fmin(func, x0=x0,
                            xtol=1.0e-15, ftol=1.0e-15, disp=False)

    model_deltaE = np.abs(func(np.array([x, y, z])) - func(x0))

    if model_deltaE > 1e-3:
        x = xm
        y = ym
        z = zm

    if ndim == 2:
        return x, y, 0
    elif ndim == 1:
        return 0, 0, z
    else:
        return x, y, z


def fit(kpts_kv, eps_k, thirdorder=False):
    import numpy.linalg as la
    A_kp = model(kpts_kv)
    if not thirdorder:
        A_kp = A_kp[:, :10]
    return la.lstsq(A_kp, eps_k, rcond=-1)


def wideMAE(masses, bt, cell_cv, erange=1e-3):
    from ase.dft.kpoints import kpoint_convert
    from ase.units import Ha
    import numpy as np

    erange = erange / Ha

    maes = []
    for i, mass in enumerate(masses['mass_u']):
        if mass is np.nan or np.isnan(mass) or mass is None:
            continue

        fit_data = masses['bs_along_emasses'][i]
        c = masses['c']
        k_kc = fit_data['kpts_kc']
        k_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=k_kc)
        e_k = fit_data['e_k'] / Ha
        assert bt == fit_data['bt']

        if bt == "vb":
            ks = np.where(np.abs(e_k - np.max(e_k)) < erange)
            assert (np.abs(e_k[ks] - np.max(e_k)) < erange).all()
            sk_kv = k_kv[ks]
        else:
            ks = np.where(np.abs(e_k - np.min(e_k)) < erange)
            sk_kv = k_kv[ks]
            assert (np.abs(e_k[ks] - np.min(e_k)) < erange).all()

        emodel_k = evalmodel(sk_kv, c, thirdorder=True)
        mae = np.mean(np.abs(emodel_k - e_k[ks])) * Ha  # eV
        maes.append(mae)

    return maes


def parsekey(k):
    if "(" not in k or ")" not in k:
        return k, k
    else:
        return k, tuple(int(x) for x in k.replace("(", "").replace(")", "").split(", "))


def iterateresults(results):
    for bk, k in map(parsekey, results):
        if type(k) != tuple:
            continue
        else:
            key_prefix = next(x for x in results[bk].keys())
            key_prefix = '_'.join(key_prefix.split('_')[:2])
            newdct = {'info': key_prefix}
            for key in results[bk].keys():
                newkey = key.replace(key_prefix + '_', '')
                newdct[newkey] = results[bk][key]
            yield k, newdct


def evalmae(cell_cv, k_kc, e_k, bt, c, erange=25e-3):
    from ase.dft.kpoints import kpoint_convert
    from ase.units import Ha, Bohr
    import numpy as np

    erange = erange / Ha

    k_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=k_kc)
    e_k = e_k.copy() / Ha

    if bt == 'vb':
        k_inds = np.where(np.abs(e_k - np.max(e_k)) < erange)[0]
        sk_kv = k_kv[k_inds, :] * Bohr
    else:
        k_inds = np.where(np.abs(e_k - np.min(e_k)) < erange)[0]
        sk_kv = k_kv[k_inds, :] * Bohr

    emodel_k = evalmodel(sk_kv, c, thirdorder=True)
    mae = np.mean(np.abs(emodel_k - e_k[k_inds])) * Ha

    return mae


def evalmare(cell_cv, k_kc, e_k, bt, c, erange=25e-3):
    from ase.dft.kpoints import kpoint_convert
    from ase.units import Ha, Bohr
    import numpy as np

    erange = erange / Ha

    k_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=k_kc)
    e_k = e_k.copy() / Ha

    if bt == 'vb':
        k_inds = np.where(np.abs(e_k - np.max(e_k)) < erange)[0]
        sk_kv = k_kv[k_inds, :] * Bohr
    else:
        k_inds = np.where(np.abs(e_k - np.min(e_k)) < erange)[0]
        sk_kv = k_kv[k_inds, :] * Bohr

    emodel_k = evalmodel(sk_kv, c, thirdorder=True)
    mare = np.mean(np.abs((emodel_k - e_k[k_inds]) / emodel_k)) * 100

    return mare


def evalparamare(fitinfo, bt, cell, k_kc, e_k):
    from ase.dft.kpoints import labels_from_kpts, kpoint_convert
    from ase.units import Bohr, Ha
    import numpy as np

    xk, _, _ = labels_from_kpts(kpts=k_kc, cell=cell)
    xk -= xk[-1] / 2.0
    kpts_kv = kpoint_convert(cell_cv=cell, skpts_kc=k_kc)
    kpts_kv *= Bohr
    emodel_k = evalmodel(kpts_kv, fitinfo, thirdorder=False) * Ha

    if bt == 'vb':
        indices = np.where(np.abs(e_k - np.max(e_k)) < 25e-3)[0]
        mean_e = np.mean(np.abs(e_k[indices] - np.max(e_k[indices])))
    else:
        indices = np.where(np.abs(e_k - np.min(e_k)) < 25e-3)[0]
        mean_e = np.mean(np.abs(e_k[indices] - np.min(e_k[indices])))

    paramare = np.mean(np.abs((emodel_k[indices] - e_k[indices]) / mean_e) * 100)

    return paramare


@command(module='asr.emasses',
         requires=['results-asr.emasses.json'],
         dependencies=['asr.emasses'],
         returns=ValidateResult)
def validate() -> ValidateResult:
    """Calculate MARE of fits over 25 meV.

    Perform a calculation for each to validate it
    over an energy range of 25 meV.

    We evaluate the MAE only along the emass directions,
    i.e. the directions shown in the plots on the website.
    """
    from asr.core import read_json
    from ase.io import read
    results = read_json('results-asr.emasses.json')
    myresults = results.copy()
    atoms = read('structure.json')

    for (sindex, kindex), data in iterateresults(results):
        # Get info on fit at this point in bandstructure
        fitinfo = data['fitcoeff']
        fitinfo2 = data['2ndOrderFit']
        bt = data['info'].split('_')[0]
        maes = []
        mares = []
        paramares = []

        for i, cutdata in enumerate(data['bzcuts']):
            k_kc = cutdata['kpts_kc']
            e_k = cutdata['e_k']
            mae = evalmae(atoms.get_cell(), k_kc, e_k, bt, fitinfo)
            maes.append(mae)
            mare = evalmare(atoms.get_cell(), k_kc, e_k, bt, fitinfo)
            mares.append(mare)

            paramare = evalparamare(fitinfo2, bt, atoms.get_cell(), k_kc, e_k)
            paramares.append(paramare)

        prefix = data['info'] + '_'
        myresults[f'({sindex}, {kindex})'][prefix + 'wideareaMAE'] = maes
        myresults[f'({sindex}, {kindex})'][prefix + 'wideareaMARE'] = mares
        myresults[f'({sindex}, {kindex})'][prefix + 'wideareaPARAMARE'] = paramares

        prefix = data['info'] + '_'
        myresults[f'({sindex}, {kindex})'][prefix + 'wideareaMAE'] = maes

    return ValidateResult(results, strict=False)


if __name__ == '__main__':
    validate.cli()
