import csv
import json
import numbers
import os
import os.path as op
import traceback
import warnings
from functools import partial
from pathlib import Path
from random import randint

import numpy as np
from numpy.linalg import eigvals

from ase import Atoms
from ase.db import connect
from ase.dft.bandgap import bandgap
from ase.io import jsonio
from ase.io import read
from ase.units import Bohr, Hartree
from ase.utils import basestring

from scipy.spatial import Delaunay

from gpaw import GPAW

from c2db import chdir, readinfo
from c2db.convex_hull import convex_hull
from c2db.em import _readmass, print_coeff
from c2db.references import formation_energy
from c2db.utils import (fermi_level, has_inversion, eigenvalues, spin_axis,
                        get_reduced_formula)


semimetal_threshold = 0.01


def stoichiometry(kvp, data, atoms, verbose):
    formula = atoms.get_chemical_formula()
    kvp['stoichiometry'] = get_reduced_formula(formula,
                                               stoichiometry=True)


def evacuum(kvp, data, atoms, verbose):
    """Try to read the vacuum level from vacuumlevels.npz,
        otherwise calculate and write it.
    """
    if op.isfile('vacuumlevels.npz'):
        d = np.load('vacuumlevels.npz')
        evac1 = float(d['evac1'])
        evac2 = float(d['evac2'])
        evacmean = (evac1 + evac2) / 2
        evacdiff = float(d['evacdiff'])
        kvp['evacmean'] = evacmean
        kvp['evac'] = evacmean
        kvp['evacdiff'] = evacdiff
        if 'dipz' in d.keys():
            kvp['dipz'] = float(d['dipz'])
    else:
        vh = GPAW('gs.gpw', txt=None).get_electrostatic_potential()
        e_vac1, e_vac2 = vh.mean(axis=0).mean(axis=0)[[0, -1]]
        kvp['evac'] = e_vac1


def gllbsc(kvp, data, atoms, verbose):
    """PBE +- SOC."""
    if not op.isfile('gllbsc.json'):
        return
    print('Collecting GLLBSC data')
    with open('gllbsc.json') as fd:
        dct = json.load(fd)

    evac = dct['evac']
    dct2 = dict(evac_gllbsc_nosoc=evac,
                deltaxc_gllbsc_nosoc=dct['deltaxc'])
    for k, v in dct.items():
        if 'gllbsc' not in k:
            continue
        if 'cbm' in k or 'vbm' in k:
            v -= evac
        dct2[k] = v
    kvp.update(dct2)


def bse(kvp, data, atoms, verbose):
    if not op.isfile('bse_pol_par.csv'):
        return
    print('Collecting BSE data')
    with open('bse_pol_par.csv') as fd:
        par = np.array([[float(x) for x in row]
                        for row in csv.reader(fd)])
    if not op.isfile('bse_pol_perp.csv'):
        return
    with open('bse_pol_perp.csv') as fd:
        per = np.array([[float(x) for x in row]
                        for row in csv.reader(fd)])

    data['bse_pol'] = {'freq': par[:, 0],
                       'par': par[:, 2],
                       'per': per[:, 2]}
    if op.isfile('eig_par.dat') and kvp.get('dir_gap', 0) > 0:
        exc = np.loadtxt('eig_par.dat')
        kvp['bse_binding'] = kvp['dir_gap'] - exc[1, 1]


def fermi(kvp, data, atoms, verbose):
    if not op.isfile('fermi_surface_soc.npz'):
        return
    print('Collecting fermi surface data')
    npz = np.load('fermi_surface_soc.npz')
    verts = npz['contours']
    invsymm = kvp.get('has_invsymm')
    if ((invsymm and kvp['magstate'] == 'NM') or
        kvp['magstate'] == 'AFM'):
        verts[:, -1] = 0  # This is the spin projection
    data['fermisurface'] = verts


def plasmafrequency(kvp, data, atoms, verbose):
    if not op.isfile('plasmafreq_tetra.npz') and \
       not op.isfile('polarizability_tetra.npz'):
        return
    if kvp.get('gap', 1) > 0.0:  # only pickup wp for metals
        return

    print('Collecting plasma frequency')
    try:
        dct = dict(np.load('plasmafreq_tetra.npz'))
    except FileNotFoundError:
        dct = dict(np.load('polarizability_tetra.npz'))
    wp2_v = eigvals(dct['plasmafreq_vv'][:2, :2])
    L = atoms.cell[2, 2] / Bohr
    plasmafreq_v = (np.sqrt(wp2_v * L / 2) *
                    Hartree * Bohr**0.5)
    kvp['plasmafrequency_x'] = plasmafreq_v[0].real
    kvp['plasmafrequency_y'] = plasmafreq_v[1].real


def absorptionspectrum(kvp, data, atoms, verbose):
    if not op.isfile('polarizability_tetra.npz'):
        return
    print('Collecting absorption data')
    dct = dict(np.load('polarizability_tetra.npz'))
    kvp['alphax'] = dct['alphax_w'][0].real
    kvp['alphay'] = dct['alphay_w'][0].real
    kvp['alphaz'] = dct['alphaz_w'][0].real
    data['absorptionspectrum'] = dct


def we_trust_gw(kvp):
    """Can we trust GW for this material?"""
    return kvp.get('gap', 0) > 0.3 and kvp.get('gap_nosoc', 0) > 0.3


def gw_gap(kvp, data, atoms, verbose):
    """gw band gaps"""
    if not os.path.isfile('gw.npz'):
        return
    if not we_trust_gw(kvp):
        return
    dct = np.load('gw.npz')
    if not float(dct['gap_gw']) > 0.0:  # only pickup gw for semiconductors
        return
    if dct['gap_gw'] < kvp.get('gap', 0.0):
        # There must be something wrong with the GW calculation
        return
    print('Collecting GW bands gap and extrema')
    evac = kvp.get('evac')
    for key in ['gap_gw', 'dir_gap_gw']:
        kvp[key] = float(dct[key])
    for key in ['cbm_gw', 'vbm_gw']:
        x = dct[key]
        if x.size == 1 and isinstance(x.tolist(), numbers.Real):
            kvp[key] = float(x) - evac


def gw_bs(kvp, data, atoms, verbose):
    """gw bandstructure and band extrema """
    from c2db.utils import get_special_2d_path
    from c2db.utils import gpw2eigs
    from c2db.gw import get_bandrange
    from ase.dft.kpoints import bandpath, parse_path_string
    if not os.path.isfile('gw.npz'):
        return
    if not we_trust_gw(kvp):
        return
    print('Collecting GW bands-structure data')
    dct = np.load('gw.npz')
    if not float(dct['gap_gw']) > 0.0:  # only pickup gw for semiconductors
        return
    evac = kvp.get('evac')
    assert dct['path'].ndim == 2
    xkreal = dct['xreal']
    data['bs_gw'] = {
        'path': dct['path'],
        'bandrange': dct['bandrange'],
        'eps_skn': dct['eps_skn'] - evac,
        'efermi': dct['efermi'] - evac,
        'epsreal_skn': dct['epsreal_skn'] - evac,
        'xkreal': xkreal}
    kvp.update(efermi_gw=dct['efermi'] - evac)
    # find high-symmetry points in xkreal
    path_str = get_special_2d_path(atoms.cell)
    path_str_list = parse_path_string(path_str)[0]
    cell = atoms.cell
    # _, _, X = bandpath(path=path_str, cell=cell, npoints=len(path_str))
    _, x, X = bandpath(path=path_str, cell=cell, npoints=len(dct['path']))
    highsym_lst = []
    for x, symbol in zip(X, path_str_list):
        delta = abs(x - xkreal)
        i = delta.argmin()
        if delta[i] < 1.0e-5:
            highsym_lst.append((i, symbol))
    # indices = [i for i, s in highsym_lst]
    data['bs_gw'].update(highsym_points=highsym_lst)
    # Try to interpolate with zero slope if pbe has it at
    # at high symm points
    gpw = 'bs.gpw'
    if not op.isfile(gpw):
        return
    n1, n2 = get_bandrange(GPAW('gs.gpw', txt=None))
    e_km, efermi = gpw2eigs('bs.gpw', soc=True)
    e_km = e_km[..., n1:n2]
    # ...


def hse_gap(kvp, data, atoms, verbose):
    evac = kvp.get('evac')
    if not op.isfile('hse_eigenvalues.npz'):
        return
    eps_skn = np.load('hse_eigenvalues.npz')['e_hse_skn']
    calc = GPAW('hse_nowfs.gpw', txt=None)
    ibzkpts = calc.get_ibz_k_points()
    efermi_nosoc = fermi_level(calc, eps_skn=eps_skn)
    gap, p1, p2 = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                             direct=True, output=None)
    if 'bs_hse' not in data:
        data['bs_hse'] = {}
    if gap:
        data['bs_hse']['kvbm_nosoc'] = ibzkpts[p1[1]]
        data['bs_hse']['kcbm_nosoc'] = ibzkpts[p2[1]]
        vbm = eps_skn[p1] - evac
        cbm = eps_skn[p2] - evac
        kvp.update(vbm_hse_nosoc=vbm, cbm_hse_nosoc=cbm,
                   dir_gap_hse_nosoc=gapd, gap_hse_nosoc=gap)

    eps = np.load('hse_eigenvalues_soc.npz')['e_hse_mk']
    eps = eps.transpose()[np.newaxis]  # e_skm, dummy spin index
    efermi = fermi_level(calc, eps_skn=eps,
                         nelectrons=calc.get_number_of_electrons() * 2)
    gap, p1, p2 = bandgap(eigenvalues=eps, efermi=efermi,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps, efermi=efermi,
                             direct=True, output=None)
    if gap:
        data['bs_hse']['kvbm'] = ibzkpts[p1[1]]
        data['bs_hse']['kcbm'] = ibzkpts[p2[1]]
        vbm = eps[p1] - evac
        cbm = eps[p2] - evac
        kvp.update(vbm_hse=vbm, cbm_hse=cbm,
                   dir_gap_hse=gapd, gap_hse=gap)
    kvp.update(efermi_hse=efermi - evac,
               efermi_hse_nosoc=efermi_nosoc - evac)


def hse(kvp, data, atoms, verbose):
    if not op.isfile('hse_bandstructure.npz'):
        return
    if op.isfile('hse_bandstructure3.npz'):
        fname = 'hse_bandstructure3.npz'
    else:
        fname = 'hse_bandstructure.npz'
    dct = dict(np.load(fname))
    if 'epsreal_skn' not in dct:
        warnings.warn('epsreal_skn missing, try and run hseinterpol again')
        return
    print('Collecting HSE bands-structure data')
    evac = kvp.get('evac')
    dct = dict(np.load(fname))
    # without soc first
    data['bs_hse'] = {
        'path': dct['path'],
        'eps_skn': dct['eps_skn'] - evac,
        # 'efermi_nosoc': efermi_nosoc - evac,
        'epsreal_skn': dct['epsreal_skn'] - evac,
        'xkreal': dct['xreal']}

    # then with soc if available
    if 'e_mk' in dct and op.isfile('hse_eigenvalues_soc.npz'):
        e_mk = dct['e_mk']  # band structure
        data['bs_hse'].update(eps_mk=e_mk - evac)


def anisotropy(kvp, data, atoms, verbose):
    if not op.isfile('anisotropy_xy.npz'):
        return
    print('Collecting anisotropy')
    dct = np.load('anisotropy_xy.npz')
    kvp['maganis_zx'] = dct['dE_zx'] * 1e3
    kvp['maganis_zy'] = dct['dE_zy'] * 1e3
    kvp['spin_orientation'] = 'xyz'[spin_axis()]


def em(kvp, data, atoms, verbose):
    """Collect effective masses and masstensor eigenvectors
    """
    print('Collecting em')
    from c2db.em import SB

    def key2str(d):
        d2 = {}
        for k in d.keys():
            if type(k) == SB:
                kstr = 'spin{}_band{}'.format(k.spin, k.band)
            else:
                kstr = str(k)
            d2[kstr] = d[k]
        return d2
    misc = data.get('miscellaneous', {})
    if data.get('effectivemass') is None:
        data['effectivemass'] = {}
    emass = data['effectivemass']
    added_mass = False
    for soc in (True, False):
        socstr = '' if soc else '_nosoc'
        if kvp.get('gap' + socstr, 0) < semimetal_threshold:
            continue
        for bt in ('vb', 'cb'):
            try:
                d = _readmass(soc=soc, bt=bt)
            except Exception:
                continue
            added_mass = True
            # add to data
            kstr = '{}{}'.format(bt, socstr)
            emass[kstr] = key2str(d)
            # add to kvp
            kstr = 'emass' if bt == 'cb' else 'hmass'
            key = d['indices'][0]     # index 0 is em for band closest Ef
            m_u = d[key]['mass_u']    # third order fit
            m2_u = d[key]['mass2_u']  # second order fit
            if bt == 'vb':
                m_u = -1 * m_u
                m2_u = -1 * m_u
            if not np.all(m_u > 0):
                warnings.warn('Wrong signs: m_u=' + str(m_u))
            if verbose:  # verbose:
                print('{} {}: {} mass={}'.format(bt, socstr, key,
                                                 np.around(m_u, 3)))
                print('fit: (3rd order)')
                print_coeff(d[key]['c'])
                print('fit (2nd order): ')
                print_coeff(d[key]['c2'])
            for u, m in enumerate(sorted(m2_u, reverse=True)):
                x = '{}{}'.format(u + 1, socstr)
                misc[kstr + '_2nd_' + x] = m
            for u, m in enumerate(sorted(m_u, reverse=True)):
                x = '{}{}'.format(u + 1, socstr)
                kvp[kstr + x] = m
    if not added_mass:
        if not data['effectivemass']:
            del data['effectivemass']
    data['miscellaneous'] = misc


def bzcut(kvp, data, atoms, verbose):
    from c2db.utils import gpw2eigs
    soc = True
    if kvp.get('gap', 0) < semimetal_threshold:
        return
    if data.get('effectivemass') is None:
        data['effective'] = {}

    for bt in ['cb', 'vb']:
        try:
            d = _readmass(soc=soc, bt=bt)
        except Exception:
            continue
        if data['effectivemass'].get(bt) is None:
            data['effectivemass'][bt] = {}
        bzcut = data['effectivemass'][bt]['bzcut_u'] = []
        sb = d['indices'][0]  # spin,band index
        mass_u = d[sb]['mass_u']
        for u, mass in enumerate(sorted(mass_u, reverse=True)):
            if abs(mass) < 1e-4:
                continue
            p = (sb.spin, sb.band, bt, u, mass, soc)
            g = 'em_bs_spin={}_band={}_bt={}_m({}){:.3e}_soc={}.gpw'.format(*p)
            if not op.isfile(g):
                warnings.warn('Not found: ' + g)
                continue
            kpts_kc = GPAW(g, txt=None).get_ibz_k_points()
            e_km, _, s_kvm = gpw2eigs(g, soc=soc, return_spin=True,
                                      optimal_spin_direction=True)
            sz_km = s_kvm[:, spin_axis(), :]
            try:
                op_scc = data['op_scc']
            except KeyError:
                from gpaw.symmetry import atoms2symmetry
                op_scc = atoms2symmetry(atoms).op_scc
            magstate = kvp['magstate']
            for idx, kpt in enumerate(kpts_kc):
                if (magstate == 'NM' and is_symmetry_protected(kpt, op_scc) or
                    magstate == 'AFM'):
                    sz_km[idx, :] = 0.0

            bzcut.append(dict(kpts_kc=kpts_kc,
                              e_dft_km=e_km,
                              sz_dft_km=sz_km))


def emexciton(kvp, data, atoms, verbose):
    if not op.isfile('em_exciton_soc.npz'):
        return
    d = np.load('em_exciton_soc.npz')
    mass_u = d['mass_u']

    for i, m in enumerate(sorted(mass_u[:2], reverse=True)):
        j = i + 1
        if 0.01 < m < 50:
            kvp['excitonmass{}'.format(j)] = float(m)


def stiffness_tensor(kvp, data, atoms, verbose):
    if not os.path.isfile('strain_quantities.npz'):
        return
    print('Collecting stiffness tensor and speed of sound.')
    d = np.load('strain_quantities.npz')
    try:
        stiffness = d['stiffness_tensor']
        speed_of_sound = d['speed_of_sound']
        kvp['c_11'] = stiffness[0, 0]
        kvp['c_22'] = stiffness[1, 1]
        kvp['c_12'] = stiffness[0, 1]
        kvp['speed_of_sound_x'] = speed_of_sound[0]
        kvp['speed_of_sound_y'] = speed_of_sound[1]
    except KeyError:
        return


def deformation_potential(kvp, data, atoms, verbose):
    name = 'strain_quantities.npz'
    if not os.path.isfile(name):
        return
    d = dict(np.load(name))
    if 'deformation_potentials_nosoc' not in d:
        return
    print('Collecting deformation potentials and mobilities.')
    for soc in ['', '_nosoc']:
        if kvp.get('gap' + soc, 0) > semimetal_threshold:
            D = d['deformation_potentials' + soc]
            kvp['D_vbm' + soc] = D[2, 0]
            kvp['D_cbm' + soc] = D[2, 1]
            data['deformation_potentials' + soc] = D


def coarsesymmetries(kvp, data, atoms, verbose):
    from gpaw.symmetry import Symmetry
    print('Calculating coarse symmetries')
    cell_cv = atoms.get_cell()
    tol = 0.01  # Tolerance for coarse symmetries
    coarsesymmetry = Symmetry(atoms.get_atomic_numbers(),
                              cell_cv, tolerance=tol, symmorphic=False,
                              rotate_aperiodic_directions=True,
                              translate_aperiodic_directions=True,
                              time_reversal=True)
    coarsesymmetry.analyze(atoms.get_scaled_positions())
    data['op_scc'] = coarsesymmetry.op_scc  # Coarse symmetry operations
    data['ft_sc'] = coarsesymmetry.ft_sc  # Fractional translations
    data['time_reversal'] = coarsesymmetry.time_reversal


def bzk(kvp, data, atoms, verbose):
    from gpaw.kpt_descriptor import to1bz
    from gpaw.bztools import get_reduced_bz
    from ase.dft.kpoints import get_special_points

    if not op.isfile('densk.gpw'):
        return

    print('Collecting IBZ k-points')
    calc = GPAW('densk.gpw', txt=None)
    new_ibz_kc = calc.get_ibz_k_points()
    new_ibz_kc = to1bz(new_ibz_kc, atoms.get_cell())
    cell_cv = atoms.get_cell()

    tol = 0.01
    op_scc = data['op_scc']
    time_reversal = data['time_reversal']
    bz_kc, ibz_kc = get_reduced_bz(cell_cv, op_scc, time_reversal,
                                   pbc_c=np.array([True, True, False]),
                                   tolerance=tol)

    # Pick the IBZ that contains the band structure path
    special_points = get_special_points(atoms.get_cell())
    tess = Delaunay([point[:2] for point in special_points.values() if
                     abs(point[2]) < 1e-10])

    op_scc = op_scc[:, :2, :2]
    if time_reversal:
        op_scc = np.concatenate([op_scc, -op_scc])

    bz_kc = bz_kc[bz_kc[:, 2]**2 < 1e-5, :2]
    ibz_kc = ibz_kc[ibz_kc[:, 2]**2 < 1e-5, :2]
    inside = 0
    for op_cc in op_scc:
        ibztmp_kc = np.dot(ibz_kc, op_cc.T)
        # How many special points are inside this IBZ?
        insidetmp = (tess.find_simplex(ibztmp_kc, tol=tol) >= 0).sum()
        if insidetmp > inside:
            ibzfinal_kc = ibztmp_kc
            inside = insidetmp

    data['ibz_kc'] = ibzfinal_kc
    data['bz_kc'] = bz_kc
    data['ibzk_pbe'] = new_ibz_kc[:, :2]


def colgap(kvp, data, atoms, verbose):
    evac = kvp.get('evac')
    misc = data.get('miscellaneous', {})
    for x, y in zip(('_soc', ''), ('', '_nosoc')):
        name = 'gap{}.npz'.format(x)
        if not op.isfile(name):
            return
        d = np.load(name)
        efermi = d['efermi']
        for socsplit in ['hsocsplit', 'esocsplit']:
            if socsplit in d:
                kvp[socsplit] = float(d[socsplit])
        if d['vbm'].tolist() is None:  # it's a metal
            print('Collecting PBE{} work-function'.format(y))
            kvp['gap' + y] = 0.0
            kvp['dir_gap' + y] = 0.0
            kvp['is_metallic' + y] = True
            kvp['work_function' + y] = evac - efermi
        else:
            print('Collecting PBE{} gap, vbm, cbm, ... data'.format(y))
            kvp['is_metallic' + y] = False
            kvp['gap' + y] = float(d['gap'])
            kvp['dir_gap' + y] = float(d['gap_dir'])
            kvp['cbm' + y] = float(d['cbm']) - evac
            kvp['vbm' + y] = float(d['vbm']) - evac
            misc['dir_vbm' + y] = float(d['vbm_dir']) - evac
            misc['dir_cbm' + y] = float(d['cbm_dir']) - evac
            if data.get('bs_pbe') is None:
                data['bs_pbe'] = {}
            tess = Delaunay(data['ibz_kc'])
            op_scc = data['op_scc']
            if data['time_reversal']:
                op_scc = np.concatenate([op_scc, -op_scc])

            # Pick point within IBZ
            point_c = data['ibzk_pbe'][d['skn1'][1]]
            mappedpoints_sc = np.dot(op_scc[:, :2, :2], point_c)
            ind = np.argwhere(tess.find_simplex(mappedpoints_sc,
                                                tol=1e-1,
                                                bruteforce=True) >= 0)[0]
            assert len(ind)
            point_c = mappedpoints_sc[ind]
            data['bs_pbe']['kvbm' + y] = np.zeros(3, float)
            data['bs_pbe']['kvbm' + y][:2] = point_c

            point_c = data['ibzk_pbe'][d['skn2'][1]]
            mappedpoints_sc = np.dot(op_scc[:, :2, :2], point_c)
            ind = np.argwhere(tess.find_simplex(mappedpoints_sc,
                                                tol=1e-1,
                                                bruteforce=True) >= 0)[0]
            assert len(ind)
            point_c = mappedpoints_sc[ind]
            data['bs_pbe']['kcbm' + y] = np.zeros(3, float)
            data['bs_pbe']['kcbm' + y][:2] = point_c
            kvp['is_dir_gap' + y] = kvp['dir_gap' + y] == kvp['gap' + y]
    data['miscellaneous'] = misc


def pdos(kvp, data, atoms, verbose):
    def old_to_new(pdos_sal):
        pdos_new_sal = {}
        symbols = atoms.get_chemical_symbols()
        for s, pdos_al in pdos_sal.items():
            for a, pdos_l in pdos_al.items():
                spec = symbols[a]
                for l, p in pdos_l.items():
                    key = '{},{},{}'.format(s, spec, l)
                    pdos_new_sal[key] = p
        return pdos_new_sal

    def getit(fname, soc):
        evac = kvp.get('evac')
        if soc:
            efermi = np.load('gap_soc.npz')['efermi'] - evac
        else:
            efermi = np.load('gap.npz')['efermi'] - evac
        with open(fname) as fd:
            dct = jsonio.decode(json.load(fd))
            pdos_sal = dct['pdos_sal']
            e = dct['energies'] - evac
        if not isinstance(list(pdos_sal.keys())[0], basestring):
            pdos_sal = old_to_new(pdos_sal)
        return {'pdos_sal': pdos_sal, 'energies': e, 'efermi': efermi}

    if op.isfile('pdos.json'):
        print('Collecting pdos w/o soc')
        data['pdos_pbe_nosoc'] = getit('pdos.json', soc=False)
    if op.isfile('pdos_soc.json'):
        print('Collecting pdos w soc')
        data['pdos_pbe'] = getit('pdos_soc.json', soc=True)

    if op.isfile('dosef_nosoc.txt'):
        with open('dosef_nosoc.txt') as fd:
            kvp['dosef_nosoc'] = float(fd.read())
    if op.isfile('dosef_soc.txt'):
        with open('dosef_soc.txt') as fd:
            kvp['dosef_soc'] = float(fd.read())


def get_started(kvp, data, skip_forces):
    folder, state = Path().cwd().parts[-2:]
    assert state in {'nm', 'fm', 'afm'}, state
    formula, _, _ = folder.partition('-')
    e_nm = read('../relax-nm.traj').get_potential_energy()
    if os.path.isfile('gs.gpw'):
        atoms = read('gs.gpw')
        calc = atoms.calc
    else:
        atoms = read('../relax-{}.traj'.format(state))
        calc = None

    for repeat in [1, 2]:
        formula = Atoms(formula * repeat).get_chemical_formula()
        if formula == atoms.get_chemical_formula():
            break  # OK
    else:
        raise ValueError('Wrong folder name: ' + folder)

    f = atoms.get_forces()
    s = atoms.get_stress()[:2]
    fmax = ((f**2).sum(1).max())**0.5
    smax = abs(s).max()

    # Allow for a bit of slack because of a small bug in our
    # modified BFGS:
    slack = 0.002
    if len(atoms) < 50 and not skip_forces:
        assert fmax < 0.01 + slack, fmax
        assert smax < 0.002, smax
    kvp['smaxinplane'] = smax

    if state == 'nm':
        assert not atoms.calc.get_spin_polarized()
        atoms.calc.results['magmom'] = 0.0
    else:
        if calc is not None:
            assert atoms.calc.get_spin_polarized()
        m = atoms.get_magnetic_moment()
        ms = atoms.get_magnetic_moments()
        if state == 'fm':
            assert abs(m) > 0.1
        else:  # afm
            assert abs(m) < 0.02 and abs(ms).max() > 0.1
    kvp['magstate'] = state.upper()
    kvp['is_magnetic'] = state != 'nm'
    kvp['cell_area'] = np.linalg.det(atoms.cell[:2, :2])
    kvp['has_invsymm'] = has_inversion(atoms)
    if state != 'nm':
        kvp['dE_NM'] = 1000 * ((atoms.get_potential_energy() - e_nm) /
                               len(atoms))
    # trying to make small negative numbers positive
    # cell = atoms.cell
    # atoms.cell = np.where(abs(cell) < 1.0e-14, 0.0, cell)
    return atoms, folder, state


def spacegroup(kvp, data, atoms, verbose):
    try:
        import spglib
    except ImportError:
        pass
    else:
        sg, number = spglib.get_spacegroup(atoms, symprec=1e-4).split()
        number = int(number[1:-1])
        print('Spacegroup:', sg, number)
        kvp['spacegroup'] = sg


def stability_level(kvp, data, atoms, verbose):

    def stability_levels(kvp):
        mineig = kvp.get('minhessianeig', np.nan)
        c_11, c_22 = kvp.get('c_11', np.nan), kvp.get('c_22', np.nan)
        if mineig < -2 or c_11 < 0 or c_22 < 0:
            dynamic_stability = 1
        elif np.isnan([mineig, c_11, c_22]).any():
            dynamic_stability = None
        elif mineig < -1e-5:
            dynamic_stability = 2
        else:
            dynamic_stability = 3
        hform = kvp.get('hform', None)
        ehull = kvp.get('ehull', None)
        if hform >= 0.2:
            thermodynamic_stability = 1
        elif hform is None or ehull is None:
            thermodynamic_stability = None
        elif ehull >= 0.2:
            thermodynamic_stability = 2
        else:
            thermodynamic_stability = 3
        return (thermodynamic_stability, dynamic_stability)

    therm, dyn = stability_levels(kvp)
    if dyn is not None:
        kvp['dynamic_stability_level'] = dyn

    if therm is not None:
        kvp['thermodynamic_stability_level'] = therm

    if verbose:
        print("Thermodynamic stability level: {}".format(therm))
        print("Dynamic stability level: {}".format(dyn))


def formationenergy(kvp, data, atoms, verbose=False):
    kvp['hform'] = formation_energy(atoms) / len(atoms)
    if verbose:
        print('Heat form:', kvp['hform'])


def phonons(kvp, data, atoms, verbose):
    from c2db.phonons import analyse
    try:
        eigs2, freqs2 = analyse(atoms, D=2)
        eigs3, freqs3 = analyse(atoms, D=3)
    except (FileNotFoundError, EOFError):
        return
    kvp['minhessianeig'] = eigs3.min()
    data['phonon_frequencies_2d'] = freqs2
    data['phonon_frequencies_3d'] = freqs3
    data['phonon_energies_2d'] = eigs2
    data['phonon_energies_3d'] = eigs3


def symmetrize_tensor(U_scc, cell_cv, tensor_vv):
    icell_cv = np.linalg.inv(cell_cv).T

    tmp_vv = np.zeros_like(tensor_vv)
    for U_cc in U_scc:
        U_vv = icell_cv.dot(U_cc).dot(cell_cv)
        tmp_vv += U_vv.dot(tensor_vv).U_vv.T
    tmp_vv /= len(U_scc)
    return tmp_vv


def piezoelectrictensor(kvp, data, atoms, verbose=False):
    deltas = [0.01, 0.005]

    P_dvvv = []
    Pclamped_dvvv = []
    spos_dvvac = []
    deltas_d = []
    
    for delta in deltas:
        deltas_d.extend([-delta, delta])
        fname = 'piezoelectrictensor-{}.json'.format(delta)
        if not op.isfile(fname):
            return

        with open(fname) as fd:
            dct = jsonio.decode(json.load(fd))

        if 'P_svvv' not in dct or 'spos_vvsac' not in dct:
            from c2db.piezoelectrictensor import piezoelectrictensor
            piezoelectrictensor(delta)
            with open(fname) as fd:
                dct = jsonio.decode(json.load(fd))

        P_dvvv.append(dct['P_svvv'][0])
        P_dvvv.append(dct['P_svvv'][1])

        Pclamped_dvvv.append(dct['Pclamped_svvv'][0])
        Pclamped_dvvv.append(dct['Pclamped_svvv'][1])
        spos_dvvac.append(dct['spos_vvsac'][:, :, 0])
        spos_dvvac.append(dct['spos_vvsac'][:, :, 1])
        
    P_dvvv = np.array(P_dvvv) / Bohr
    Pclamped_dvvv = np.array(Pclamped_dvvv) / Bohr
    deltas_d = np.array(deltas_d)
    spos_dvvac = np.array(spos_dvvac)
    
    # Sort after delta
    inds = np.argsort(deltas_d)
    deltas_d = deltas_d[inds]
    Pclamped_dvvv = Pclamped_dvvv[inds, :, :, :]
    P_dvvv = P_dvvv[inds, :, :, :]
    spos_dvvac = spos_dvvac[inds]
    
    cell_cv = atoms.get_cell()
    icell_cv = np.linalg.inv(cell_cv).T
    area = kvp['cell_area']
    
    # Take modulo a pol quantum
    phase_dvvc = (-2 * np.pi * area *
                  np.dot(P_dvvv.transpose(0, 2, 3, 1), icell_cv.T))
    phaseclamped_dvvc = (-2 * np.pi * area *
                         np.dot(Pclamped_dvvv.transpose(0, 2, 3, 1),
                                icell_cv.T))

    ndeltas = len(deltas_d)
    for i in range(1, ndeltas):
        dphase_vvc = phase_dvvc[i] - phase_dvvc[i - 1]
        mod_vvc = np.round(dphase_vvc / (2 * np.pi)) * 2 * np.pi
        phase_dvvc[i] -= mod_vvc

        dphaseclamped_vvc = phaseclamped_dvvc[i] - phaseclamped_dvvc[i - 1]
        mod_vvc = np.round(dphaseclamped_vvc / (2 * np.pi)) * 2 * np.pi
        phaseclamped_dvvc[i] -= mod_vvc

        dspos_vvac = spos_dvvac[i] - spos_dvvac[i - 1]
        mod_vvac = np.round(dspos_vvac)
        spos_dvvac[i] -= mod_vvac
    P_dvvv = (-np.dot(phase_dvvc, cell_cv).transpose(0, 3, 1, 2) /
              (2 * np.pi * area))

    Pclamped_dvvv = (-np.dot(phaseclamped_dvvc,
                             cell_cv).transpose(0, 3, 1, 2) /
                     (2 * np.pi * area))
    
    Ptmp_dv = P_dvvv.reshape(len(deltas_d), -1)
    A = np.vstack([deltas_d, np.ones(len(deltas_d))]).T
    B, R, rank, s = np.linalg.lstsq(A, Ptmp_dv, rcond=None)
    e_vvv = B[0, :].reshape(3, 3, 3)
    R_vvv = R
    
    Ptmp_dv = Pclamped_dvvv.reshape(len(deltas_d), -1)
    A = np.vstack([deltas_d, np.ones(len(deltas_d))]).T
    B, R, rank, s = np.linalg.lstsq(A, Ptmp_dv, rcond=None)
    R0_vvv = R
    e0_vvv = B[0, :].reshape(3, 3, 3)

    # if np.any(R_vvv) < 0.5:
    #     return

    # if np.any(R0_vvv) < 0.5:
    #     return

    print('Collecting piezoelectric tensor')
    
    # Voigt notation
    e0_ij = e0_vvv[:, [0, 1, 2, 1, 0, 0],
                   [0, 1, 2, 2, 2, 1]]
    e_ij = e_vvv[:, [0, 1, 2, 1, 0, 0],
                 [0, 1, 2, 2, 2, 1]]
    
    # Add key value pairs: e_11, e_12, ..., e_21, ..., e_36
    for i in range(3):
        for j in range(6):
            kvp['e0_{}{}'.format(i + 1, j + 1)] = e0_ij[i, j]
            kvp['e_{}{}'.format(i + 1, j + 1)] = e_ij[i, j]

    kvp['maxpiezo'] = np.max(e_ij)
    # Store raw data
    data['e0_vvv'] = e0_vvv
    data['e_vvv'] = e_vvv

    data['piezodata'] = {'deltas_d': deltas_d,
                         'clamped': Pclamped_dvvv,
                         'true': P_dvvv,
                         'spos_dvvac': spos_dvvac,
                         'eqspos_ac': atoms.get_scaled_positions()}


def uid(kvp, data, atoms, verbose):
    """Set temporary uid.

    Will be changed later once we know the prototype.
    """
    formula = atoms.get_chemical_formula()
    uid = '{}-X-{}-{}'.format(formula,
                              kvp['magstate'],
                              randint(2, 9999999))
    kvp['uid'] = uid


def collect(info, db, verbose=False, skip_forces=False, references=None):
    kvp = {}
    data = {}
    errors = []
    if op.isfile('duplicate'):
        return errors
    atoms, folder, state = get_started(kvp, data, skip_forces)
    kvp['folder'] = str(Path().cwd())

    prototype = info.get('prototype')
    cls = info.get('class')
    if prototype:
        kvp['prototype'] = prototype
    if cls:
        kvp['class'] = cls

    from c2db.dimensionality import is_2d
    connected = is_2d(atoms=atoms)
    if prototype != 'PbA2I4' and not connected:
        return errors
    steps = [uid,
             stoichiometry,
             formationenergy,
             anisotropy,
             stiffness_tensor,
             phonons,
             coarsesymmetries,
             partial(convex_hull, references=references),
             stability_level,
             spacegroup]
    if op.isfile('gs.gpw'):
        # Also collect properties
        steps += [evacuum,
                  absorptionspectrum,
                  bs,
                  bzk,
                  colgap,
                  bse,
                  em,
                  bzcut,
                  emexciton,
                  deformation_potential,
                  fermi,
                  gllbsc,
                  gw_gap,
                  gw_bs,
                  hse_gap,
                  hse,
                  pdos,
                  plasmafrequency,
                  piezoelectrictensor]

    from pathlib import Path
    from importlib import import_module
    
    pathlist = Path(__file__).parent.glob('*.py')
    for path in pathlist:
        name = path.with_suffix('').name
        module = import_module('.recipies.' + name, package='mcr')
        try:
            steps.append(module.collect_data)
        except AttributeError:
            continue

    for step in steps:
        try:
            step(kvp=kvp, data=data, atoms=atoms, verbose=verbose)
        except KeyboardInterrupt:
            raise
        except Exception as x:
            error = '{}: {}'.format(x.__class__.__name__, x)
            tb = traceback.format_exc()
            errors.append((folder + '/' + state, error, tb))
    if db is not None:
        db.write(atoms, data=data, **kvp)
    return errors


def collect_all(**kwargs):
    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), 'database.db')
    if not kwargs.dry_run:
        db = connect(dbname)

    errors = []
    n = len(kwargs.folder)
    for i, folder in enumerate(kwargs.folder):
        if not os.path.isdir(folder):
            continue
        with chdir(folder):
            print(folder, end=': ')
            info = readinfo('..')
            prototype = info.get('prototype')
            if prototype:
                print('{:3}/{:3} [{}]'.format(i + 1, n, prototype), flush=True)
            try:
                if kwargs.references:
                    kwargs.references = Path(kwargs.references).resolve()
                errors2 = collect(info, db, verbose=kwargs.verbose,
                                  skip_forces=kwargs.skipforces,
                                  references=kwargs.references)
            except KeyboardInterrupt:
                break
            except Exception as x:
                error = '{}: {}'.format(x.__class__.__name__, x)
                tb = traceback.format_exc()
                print(error)
                errors.append((folder, error, tb))
            else:
                errors.extend(errors2)

    if errors:
        print('Errors:')
        for error in errors:
            print('{}\n{}: {}\n{}'.format('=' * 77, *error))


def get_parser():
    import argparse
    description = 'Collect data from folders and put into ase.database'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('folder', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--skipforces', action='store_true')
    parser.add_argument('-n', '--dry-run', action='store_true')
    parser.add_argument('-r', '--references',
                        help='Path to 1 and 2 component reference database.')
    parser.add_argument('-f', '--filename', default='database.db',
                        help='Filename for database')
    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop('func', None)
    collect_all(**kwargs)


if __name__ == '__main__':
    main()
