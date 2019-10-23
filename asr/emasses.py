from asr.core import command, option

# TODO Resources?
@command('asr.emasses',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate', 'asr.structureinfo'],
         creates=['em_circle_vb_nosoc.gpw', 'em_circle_cb_nosoc.gpw',
                   'em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw'])
@option('--gpwfilename', type=str,
        help='GS Filename')
def refine(gpwfilename='gs.gpw'):
    '''
    Take a bandstructure and calculate more kpts around
    the vbm and cbm
    '''
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    import os.path
    import traceback
    socs = [True, False]

    for soc in socs:
        eigenvalues, efermi = gpw2eigs(gpw=gpwfilename, soc=soc,
                                       optimal_spin_direction=True)
        gap, _, _ = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                            output=None)
        for bt in ['vb', 'cb']:
            name = get_name(soc=soc, bt=bt)
            gpw2 = name + '.gpw'
            
            if not gap > 0:
                from gpaw import GPAW
                calc = GPAW(gpwfilename, txt=None)
                calc.write(gpw2)
                continue
            if os.path.exists(gpw2):
                continue
            nonsc_sphere(gpw=gpwfilename, soc=soc, bandtype=bt)
            
            
def get_name(soc, bt):
    return 'em_circle_{}_{}'.format(bt, ['nosoc', 'soc'][soc])

def nonsc_sphere(gpw='gs.gpw', soc=False, bandtype=None):
    """non sc calculation based for kpts in a sphere around the
        valence band maximum and conduction band minimum.
        writes the files:
            em_circle_vb_soc.gpw
            em_circle_cb_soc.gpw
            em_circle_vb_nosoc.gpw
            em_circle_cb_nosoc.gpw
        Parameters:
            gpw: str
                gpw filename
            soc: bool
                spinorbit coupling
            bandtype: None or 'cb' or 'vb'
                which bandtype do we do calculations for, if None is done for
                for both cb and vb

    """
    from gpaw import GPAW, PW
    import numpy as np
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    calc = GPAW(gpw, txt=None)
    ndim = calc.atoms.pbc.sum()
    # Check that 1D: Only x-axis, 2D: Only x- and y-axis
    #assert np.allclose(calc.atoms.pbc[ndim:], 0)

    k_kc = calc.get_ibz_k_points()
    cell_cv = calc.atoms.get_cell()
    kcirc_kc = kptsinsphere(cell_cv, dimensionality=ndim)

    e_skn, efermi = gpw2eigs(gpw, soc=soc, optimal_spin_direction=True)
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]

    _, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn, efermi=efermi,
                                            output=None)

    k1_c = k_kc[k1]
    k2_c = k_kc[k2]

    bandtypes, ks = get_bt_ks(bandtype, k1_c, k2_c)
    
    for bt, k_c in zip(bandtypes, ks):
        name = get_name(soc=soc, bt=bt)
        calc.set(kpts=kcirc_kc + k_c,
                 symmetry='off',
                 txt=name + '.txt')
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

    # print(kpts_kv)

    # a = np.linspace(-1, 1, npoints)
    # X, Y, Z = np.meshgrid(a, a, a)
    # indices = X**2 + Y**2 + Z**2 <= 1.0
    # sh = X.shape
    # x, y, z = X[indices], Y[indices], Z[indices]
    # kpts_kv = np.vstack([x, y * yfactor, z * zfactor]).T

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


def webpanel(row, key_descriptions):
    from asr.browser import table

    t = table(row, 'Postprocessing',
              ['cb_emass', 'vb_emass'],
              key_descriptions)
    
    panel = ('Effective masses', [[t]])
    return panel, None


# TODO Resources?
@command('asr.emasses',
         requires=['em_circle_vb_nosoc.gpw', 'em_circle_cb_nosoc.gpw',
                   'em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw', 'gs.gpw'],
         dependencies=['asr.emasses@refine', 'asr.gs@calculate'],
         webpanel=webpanel)
@option('--gpwfilename', type=str,
         help='GS Filename')
def main(gpwfilename='gs.gpw'):
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    import os.path
    import traceback
    socs = [True, False]

    good_results = {}
    for soc in socs:
        eigenvalues, efermi = gpw2eigs(gpw=gpwfilename, soc=soc,
                                       optimal_spin_direction=True)
        gap, _, _ = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                            output=None)
        if not gap > 0:
            continue
        for bt in ['vb', 'cb']:
            name = get_name(soc=soc, bt=bt)
            gpw2 = name + '.gpw'
            try:
                masses = embands(gpw2,
                                 soc=soc,
                                 bandtype=bt,
                                 efermi=efermi)

                # This function modifies the last argument
                unpack_masses(masses, soc, bt, good_results)
            except ValueError:
                tb = traceback.format_exc()
                print(gpw2 + ':\n' + '=' * len(gpw2) + '\n', tb)
            else:
                _savemass(soc=soc, bt=bt, mass=masses)
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
    for ind in masses['indices']:
        out_dict = masses[ind]
        index = str(ind)
        socpre = 'soc' if soc else 'nosoc'
        prefix = bt + '_' + socpre + '_'

        results_dict[index] = {}

        results_dict[index][prefix + 'effmass_dir1'] = out_dict['mass_u'][0]
        results_dict[index][prefix + 'effmass_dir2'] = out_dict['mass_u'][1]
        results_dict[index][prefix + 'effmass_dir3'] = out_dict['mass_u'][2]
        results_dict[index][prefix + 'eigenvectors_vdir1'] = out_dict['eigenvectors_vu'][:, 0]
        results_dict[index][prefix + 'eigenvectors_vdir2'] = out_dict['eigenvectors_vu'][:, 1]
        results_dict[index][prefix + 'eigenvectors_vdir3'] = out_dict['eigenvectors_vu'][:, 2]
        results_dict[index][prefix + 'spin'] = ind[0]
        results_dict[index][prefix + 'bandindex'] = ind[1]
        results_dict[index][prefix + 'kpt_v'] = out_dict['ke_v']
        results_dict[index][prefix + 'fitcoeff'] = out_dict['c']
        results_dict[index][prefix + 'mass_u'] = out_dict['mass_u']
        results_dict[index][prefix + 'bzcuts'] = out_dict['bs_along_emasses']


# Calc energies along eff mass eig vecs

def embands(gpw, soc, bandtype, efermi=None, delta=0.1):
    """effective masses for bands within delta of extrema
    Parameters:
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
    calc = GPAW(gpw, txt=None)

    e_skn, efermi2 = gpw2eigs(gpw, soc=soc, optimal_spin_direction=True)
    if efermi is None:
        efermi = efermi2
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]
    vb_ind, cb_ind = get_vb_cb_indices(e_skn=e_skn, efermi=efermi, delta=delta)

    indices = vb_ind if bandtype == 'vb' else cb_ind
    atoms = calc.get_atoms()
    cell_cv = atoms.get_cell()
    ibz_kc = calc.get_ibz_k_points()
       
    ibz_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ibz_kc)
    masses = {'indices': indices}
    for b in indices:
        e_k = e_skn[b[0], :, b[1]]
        masses[b] = em(kpts_kv=ibz_kv * Bohr,
                       eps_k=e_k / Hartree, bandtype=bandtype)

        masses[b]['bs_along_emasses'] = calculate_bs_along_emass_vecs(masses[b], soc, bandtype, calc)

    return masses

def calculate_bs_along_emass_vecs(masses_dict, soc, bt, calc, erange=500e-3, npoints=91):
    # out = dict(mass_u=masses, eigenvectors_vu=vecs,
    #            ke_v=kmax,
    #            c=c,
    #            r=r)
    from ase.units import Hartree, Bohr
    from ase.dft.kpoints import kpoint_convert
    from asr.utils.gpw2eigs import gpw2eigs
    from asr.utils.spinutils import spin_axis
    from asr.utils.symmetry import is_symmetry_protected
    import numpy as np
    from gpaw.mpi import rank
    cell_cv = calc.get_atoms().get_cell()

    results_dicts = []
    for u, mass in enumerate(masses_dict['mass_u']):
        # embzcut stuff
        kmax = np.sqrt(2 * abs(mass) * erange / Hartree)
        kd_v = masses_dict['eigenvectors_vu'][:, u]
        k_kv = (np.linspace(-1, 1, npoints) * kmax * kd_v.reshape(3, 1)).T
        k_kv += masses_dict['ke_v']
        k_kv /= Bohr
        k_kc = kpoint_convert(cell_cv=cell_cv, ckpts_kv=k_kv)
        atoms = calc.get_atoms()
        skip_it = False
        for i, pb in enumerate(atoms.pbc):
            if not pb and not np.allclose(k_kc[:, i], 0):
                results_dicts.append(dict())
                skip_it = True
                break
        if skip_it:
            continue
        
        calc.set(kpts=k_kc, symmetry='off', txt=None)
        atoms.get_potential_energy()
        name = 'temp.gpw'
        calc.write(name)

        # Start of collect.py stuff
        e_km, _, s_kvm = gpw2eigs(name, soc=soc, return_spin=True,
                       optimal_spin_direction=True)

        sz_km = s_kvm[:, spin_axis(), :]
        from gpaw.symmetry import atoms2symmetry
        op_scc = atoms2symmetry(calc.get_atoms()).op_scc # Ask Morten, get it from somewhere?
        magstate = 'NM' # Ask Morten, get it from where?
        for idx, kpt in enumerate(k_kc):
            if (magstate == 'NM' and is_symmetry_protected(kpt, op_scc) or
                magstate == 'AFM'):
                sz_km[idx, :] = 0.0

        # Start of custom.py stuff
        # xb is (v/c)b dict from data["effectivemass"]["(v/c)b"]
        ## It contains 
        #b_u is list of kpts, es, szs for eff mass dirs (u) for given bt

        # custom.py processing requires other data from eff mass calculation
        
        results_dicts.append(dict(kpts_kc=k_kc,
                                  e_dft_km=e_km,
                                  sz_dft_km=sz_km))

def get_vb_cb_indices(e_skn, efermi, delta):
    """
    find CB and VB within a distance of delta of the CB and VB extrema
    Parameters:
        e_skn: (ns, nk, nb)-shape ndarray
            eigenvalues
        efermi: float
            fermi level
        delta: float
            bands within delta of the extrema are included
    Returns:
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


def em(kpts_kv, eps_k, bandtype=None):
    """
    Parameters:
        kpts_kv: (nk, 3)-shape ndarray
            k-points in cartesian coordinates (in units of 1 / Bohr)
        eps_k: (nk,)-shape ndarray
            eigenvalues (in units of Hartree)
    Returns:
        out: dct
            - effective masses in units of m_e
            - eigenvectors in cartesian coordinates
            - k-pot extremum in cartesian coordinates (units of 1 / Bohr)

    """
    import numpy as np
    c, r, rank, s, = fit(kpts_kv, eps_k, thirdorder=False)
    fxx = 2 * c[0]
    fyy = 2 * c[1]
    fzz = 2 * c[2]
    fxy = c[3]
    fxz = c[4]
    fyz = c[5]
    fx = c[6]
    fy = c[7]
    fz = c[8]

    # Get min/max location from 2nd order fit to evalulate
    # the second order derivative in the third order fit
    xm, ym, zm = get_2nd_order_extremum(c)
    ke2_v = np.array([xm, ym, zm])

    c3, r3, rank3, s3 = fit(kpts_kv, eps_k, thirdorder=True)

    f3xx, f3yy, f3zz, f3xy, f3xz, f3yz, f3x, f3y, f3z, f30, f3xxx, f3yyy, f3zzz, f3xxy, f3xxz, f3yyx, f3yyz, f3zzx, f3zzy, f3xyz = c3
    
    extremum_type = get_extremum_type(fxx, fyy, fzz, fxy, fxz, fyz)
    xm, ym, zm = get_3rd_order_extremum(xm, ym, zm, c3, extremum_type)
    ke_v = np.array([xm, ym, zm])


    d3xx = 2 * f3xx + 6 * f3xxx * xm + 2 * f3xxy * ym + 2 * f3xxz * zm 
    d3yy = 2 * f3yy + 6 * f3yyy * ym + 2 * f3yyx * xm + 2 * f3yyz * zm
    d3zz = 2 * f3zz + 6 * f3zzz * zm + 2 * f3zzx * xm + 2 * f3zzy * ym
    d3xy = f3xy + 2 * f3xxy * xm + 2 * f3yyx * ym + f3xyz * zm
    d3xz = f3xz + 2 * f3xxz * xm + 2 * f3zzx * zm + f3xyz * ym
    d3yz = f3yz + 2 * f3yyz * ym + 2 * f3zzy * zm + f3xyz * xm
    
    hessian3 = np.array([[d3xx, d3xy, d3xz],
                         [d3xy, d3yy, d3yz],
                         [d3xz, d3yz, d3zz]])
    
    v3_n, w3_vn = np.linalg.eigh(hessian3)

    # This commented out code is needed for further
    # refinement of the effective mass calculation
    # def get_bt(fxx, fyy, fzz, fxy, fxz, fyz):
    #    hessian = np.array([[fxx, fxy, fxz],
    # [fxy, fyy, fyz], [fxz, fyz, fzz]])
    #    detH = np.linalg.det(hessian)
    #    if detH < 0:
    #        bandtype = 'saddlepoint'
    #    elif fxx < 0 and fyy < 0 and fzz < 0:
    #        bandtype = 'vb'
    #    elif fxx > 0 and fyy > 0 and fzz > 0:
    #        bandtype = 'cb'
    #    else:
    #        raise ValueError("Bandtype could not be found.
    # Hessian had determinant: {}"
    # .format(detH))
    #    return bandtype
    # if bandtype is None:
    #    bandtype = get_bt(fxx, fyy, fzz, fxy, fxz, fyz)
    hessian = np.array([[fxx, fxy, fxz], [fxy, fyy, fyz], [fxz, fyz, fzz]])
    v2_n, vecs = np.linalg.eigh(hessian)
        

    out = dict(mass_u=1 / v3_n,
               eigenvectors_vu=w3_vn,
               ke_v=ke_v,
               c=c3,
               r=r3,
               mass2_u=1 / v2_n,
               eigenvectors2_vu=vecs,
               ke2_v=ke2_v,
               c2=c,
               r2=r)

    return out

def get_extremum_type(dxx, dyy, dzz, dxy, dxz, dyz):
    # Input: 2nd order derivatives at the extremum point
    import numpy as np
    hessian = np.array([[dxx, dxy, dxz],
                        [dxy, dyy, dyz],
                        [dxz, dyz, dzz]])
    vals, vecs = np.linalg.eigh(hessian)
    saddlepoint = not (np.sign(vals[0]) == np.sign(vals[1]) and np.sign(vals[0]) == np.sign(vals[2]))

    if saddlepoint:
        etype = 'saddlepoint'
    elif (vals < 0).all():
        etype = 'max'
    elif (vals > 0).all():
        etype = 'min'
    else:
        raise ValueError('Extremum type could not be determined for hessian: {}'.format(hessian))
    return etype

def get_2nd_order_extremum(c):
    import numpy as np
    # fit is 
    # fxx x^2 + fyy y^2 + fzz z^2 + fxy xy + fxz xz + fyz yz + fx x + fy y + fz z + f0
    assert len(c) == 10
    fxx, fyy, fzz, fxy, fxz, fyz, fx, fy, fz, f0 = c
    
    ma = np.array([[2 * fxx, fxy, fxz],
                   [fxy, 2 * fyy, fyz],
                   [fxz, fyz, 2 * fzz]])

    v = np.array([-fx, -fy, -fz])

    min_pos = np.linalg.solve(ma, v)

    return min_pos

def get_3rd_order_extremum(xm, ym, zm, c, extremum_type):
    import numpy as np
    from scipy import optimize
    # We want to use a minimization function from scipy
    # so if the extremum type is a 'max' we need to multiply
    # the function by - 1
    assert len(c) == 20

    def get_v(kpts):
        k = np.asarray(kpts)
        if k.ndim == 1:
            k = k[np.newaxis]
        return model(k)

    if extremum_type == 'max':
        func = lambda v: -1 * np.dot(get_v(v), c)
    else:
        func = lambda v: np.dot(get_v(v), c)

    x0 = np.array([xm, ym, zm])
    x, y, z = optimize.fmin(func, x0=x0, xtol=1.0e-15, ftol=1.0e-15, disp=False)
    return x, y, z

def fit(kpts_kv, eps_k, thirdorder=False):
    import numpy.linalg as la
    A_kp = model(kpts_kv)
    if not thirdorder:
        A_kp = A_kp[:, :10]
    return la.lstsq(A_kp, eps_k, rcond=-1)


def model(kpts_kv):
    """ simple third order model
        Parameters:
            kpts_kv: (nk, 3)-shape ndarray
                units of (1 / Bohr)
    """
    import numpy as np
    k_kx, k_ky, k_kz = kpts_kv[:, 0], kpts_kv[:, 1], kpts_kv[:, 2]

    ones = np.ones(len(k_kx))

    A_dp = np.array([k_kx**2,
                     k_ky**2,
                     k_kz**2,
                     k_kx * k_ky,
                     k_kx * k_kz,
                     k_ky * k_kz,
                     k_kx,
                     k_ky,
                     k_kz,
                     ones,
                     k_kx**3,
                     k_ky**3,
                     k_kz**3,
                     k_kx**2 * k_ky,
                     k_kx**2 * k_kz,
                     k_ky**2 * k_kx,
                     k_ky**2 * k_kz,
                     k_kz**2 * k_kx,
                     k_kz**2 * k_ky,
                     k_kx * k_ky * k_kz]).T

    return A_dp


def _savemass(soc, bt, mass):
    from ase.parallel import world
    import numpy as np
    fname = get_name(soc, bt) + '.npz'
    if world.rank == 0:
        mass2 = {}
        for k, v in mass.items():
            if type(k) == tuple:
                mass2[k] = v
            elif k == 'indices':
                mass2[k] = [tuple(vi) for vi in v]
            else:
                mass2[k] = v
        with open(fname, 'wb') as f:
            np.savez(f, data=mass2)
    world.barrier()


def _readmass(soc, bt):
    import numpy as np
    fname = get_name(soc=soc, bt=bt) + '.npz'
    with open(fname, 'rb') as f:
        dct = dict(np.load(f))['data'].tolist()
    return dct


def collect_data(atoms):
    from pathlib import Path
    all_data = {}
    kvp = {}
    key_descriptions = {}
    if not list(Path('.').glob('em_circle_*.npz')):
        return {}, {}, {}

    for soc in [True, False]:
        keyname = 'soc' if soc else 'nosoc'
        data = {}
        for bt in ['cb', 'vb']:
            temp = _readmass(soc, bt)
            for key, val in temp.items():
                if key == 'indices':
                    continue
                else:
                    data[bt] = val
        all_data[keyname] = data

    descs = [('Conduction Band emasses',
              'Effective masses for conduction band', '-'),
             ('Valence Band emasses',
              'Effective masses for conduction band', '-'),
             ('Conduction Band emasses with SOC',
              'Effective masses with spin-orbit coupling for conduction band',
              '-'),
             ('Valence Band emasses with SOC',
              'Effective masses with spin-orbit coupling for valence band',
              '-')
             ]

    for socname, socdata in all_data.items():
        soc = socname == 'soc'

        def namemod(n):
            return n + '_soc' if soc else n
        for bt, btdata in socdata.items():
            key = bt + '_emass'
            key = namemod(key)
            kvp[key] = btdata

            if soc:
                key_descriptions[key] = descs[0] if bt == 'cb' else descs[1]
            else:
                key_descriptions[key] = descs[2] if bt == 'cb' else descs[3]
                
    return kvp, key_descriptions, all_data




# def webpanel(row, key_descriptions):

#     from asr.browser import fig
#     add_nosoc = ['D_vbm', 'D_cbm', 'is_metallic', 'is_dir_gap',
#                  'emass1', 'emass2', 'hmass1', 'hmass2', 'work_function']

#     def nosoc_update(string):
#         if string.endswith(')'):
#             return string[:-1] + ', no SOC)'
#         else:
#             return string + ' (no SOC)'

#     for key in add_nosoc:
#         s, l, units = key_descriptions[key]
#         if l:
#             key_descriptions[key + "_nosoc"] = (s, nosoc_update(l), units)
#         else:
#             key_descriptions[key + "_nosoc"] = (nosoc_update(s), l, units)

#     panel = ('Effective masses (PBE)',
#              [[fig('pbe-bzcut-cb-bs.png'), fig('pbe-bzcut-vb-bs.png')],
#               emtables(row)])

#     return panel


if __name__ == '__main__':
    main.cli()
