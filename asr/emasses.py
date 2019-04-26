def emtables(row):
    if row.data.get('effectivemass') is None:
        return [None, None]
    unit = 'm<sub>e</sub>'
    tables = []
    for bt in ['cb', 'vb']:
        dct = row.data.effectivemass.get(bt)
        if dct is None:
            tables.append(None)
            continue
        if bt == 'cb':
            title = 'Electron effective mass'
        else:
            title = 'Hole effective mass'
        keys = [k for k in dct.keys() if 'spin' in k and 'band' in k]
        rows = []
        for i, k in enumerate(keys):
            emdata = dct[k]
            m_u = emdata['mass_u']
            if bt == 'vb':
                m_u = -m_u
            if i == 0:
                desc = '{}'.format(bt.upper())
            else:
                sgn = ' + ' if bt == 'cb' else ' - '
                desc = '{}{}{}'.format(bt.upper(), sgn, i)
            for u, m in enumerate(sorted(m_u, reverse=True)):
                if 0.001 < m < 100:  # masses should be reasonable
                    desc1 = ', direction {}'.format(u + 1)
                    rows.append([desc + desc1,
                                 '{:.2f} {}'.format(m, unit)])
        tables.append({'type': 'table',
                       'header': [title, ''],
                       'rows': rows})
    return tables


# def webpanel(row, key_descriptions):

#     from asr.custom import fig
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




from asr.utils import click, update_defaults, get_start_parameters
params = get_start_parameters()
defaults = {}


#@click.options('--something', type=str, help='HELP', default='smth')
@click.command()
@update_defaults('asr.emasses', defaults)
@click.options('--gpwfilename', type=str, help='Filename of GS calculation', default='gs.gpw')
def main(gpwfilename):
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    import os.path
    import traceback
    socs = [True, False]

    for soc in socs:
        eigenvalues, efermi = gpw2eigs(gpw=gpw, soc=soc,
                                       optimal_spin_direction=True)
        gap, _, _ = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                            output=None)
        if not gap > 0:
            continue
        for bt in ['vb', 'cb']:
            name = get_name(soc=soc, bt=bt)
            gpw2 = name + '.gpw'
            if not os.path.isfile(gpw2):
                nonsc_circle(gpw=gpwfilename, soc=soc, bandtype=bt)
            try:
                masses = embands(gpw2,
                                 soc=soc,
                                 bandtype=bt,
                                 efermi=efermi)
            except ValueError:
                tb = traceback.format_exc()
                print(gpw2 + ':\n' + '=' *len(gpw2) + '\n', tb)
            else:
                _savemass(soc=soc, bt=bt, mass=masses)
            
        

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
    from gpaw import GPAW
    import numpy as np
    calc = GPAW(gpw, txt=None)
    ndim = calc.atoms.pbc.sum()
    assert np.allclose(calc.atoms.pbc[ndim:], 0) #Check that 1D: Only x-axis, 2D: Only x- and y-axis
    if ndim == 1:
        raise NotImplementedError("Recipe not implemented for 1D")

    k_kc = calc.get_ibz_k_points()
    cell_cv = calc.atoms.get_cell()
    kcirc_kc = kptsinsphere(cell_cv)
    e_skn, efermi = gpw2eigs(gpw, soc=soc, optimal_spin_direction=True)
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]
    _, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn, efermi=efermi,
                                            output=None)
    k1_c = k_kc[k1]
    k2_c = k_kc[k2]

    if bandtype is None:
        bandtypes = ('vb', 'cb')
        ks = (k1_c, k2_c)
    elif bandtype == 'vb':
        bandtypes = ('vb',)
        ks = (k1_c, )
    elif bandtype == 'cb':
        bandtypes = ('cb', )
        ks = (k2_c, )
    
    for bt, k_c in zip(bandtypes, ks):
        name = get_name(soc=soc, bt=bt)
        calc.set(kpts=kcirc_kc + k_c,
                 symmetry='off',
                 txt=name + '.txt')
        atoms = calc.get_atoms()
        atoms.get_potential_energy()
        calc.write(name + '.gpw')


def kptsinsphere(cell_cv, npoints=9, erange=1e-3, m=1.0, 2d=True):
    import numpy as np
    from ase.units import Hartree, Bohr
    from ase.dft.kpoints import kpoint_convert
    if 2d:
        #This factor is used to kill contribution from z-coordinates in 2D case
        zfactor = 0
    else:
        zfactor = 1

    a = np.linspace(-1, 1, npoints)
    X, Y, Z = np.meshgrid(a, a, a)
    indices = X**2 + Y**2 + zfactor*Z**2 <= 1
    x, y, z = X[indices], Y[indices], Z[indices]
    kpts_kv = np.vstack([x, y, z*zfactor]).T
    kr = np.sqrt(2*m*erange/Hartree)
    kpts_kv *= kr
    kpts_kv /= Bohr
    kpts_kc = kpoint_convert(cell_cv=cell_cv, ckpts_kv=kpts_kv)
    return kpts_kc


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
        masses[b] = em(kpts_kv=ibz_kv*Bohr,
                       eps_k=e_k/Hartree, bandtype=bandtype)
    return masses


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
     gap, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn, efermi=efermi, output=None)
     
     if not gap > 0:
         raise ValueError('Band gap is zero')
     
     cbm = e_skn[s2, k2, n2]
     vbm = e_skn[s1, k1, n1]

     cb_sn = e_skn[:, k2, n2:]
     vb_sn = e_skn[:, k1, :n1+1]
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
    #Do a second order fit to get started
    c, r, rank, s, = fit(kpts_kv, eps_k, thirdorder=False)
    fxx = 2*c[0]
    fyy = 2*c[1]
    fzz = 2*c[2]
    fxy = c[3]
    fxz = c[4]
    fyz = c[5]
    fx = c[6]
    fy = c[7]
    fz = c[8]
    
    ##This commented out code is needed for further refinement of the effective mass calculation
    #def get_bt(fxx, fyy, fzz, fxy, fxz, fyz):
    #    hessian = np.array([[fxx, fxy, fxz], [fxy, fyy, fyz], [fxz, fyz, fzz]])
    #    detH = np.linalg.det(hessian)
    #    if detH < 0:
    #        bandtype = 'saddlepoint'
    #    elif fxx < 0 and fyy < 0 and fzz < 0:
    #        bandtype = 'vb'
    #    elif fxx > 0 and fyy > 0 and fzz > 0:
    #        bandtype = 'cb'
    #    else:
    #        raise ValueError("Bandtype could not be found. Hessian had determinant: {}".format(detH))
    #    return bandtype
    #if bandtype is None:
    #    bandtype = get_bt(fxx, fyy, fzz, fxy, fxz, fyz)
    hessian = np.array([[fxx, fxy, fxz], [fxy, fyy, fyz], [fxz, fyz, fzz]])
    masses, vecs = np.linalg.eigh(hessian)
    
    #Calculate extremum point
    A = np.array([fx, fy, fz])
    kmax = -0.5*A.dot(np.linalg.inv(hessian))

    out = dict(mass_u=masses, eigenvectors_vu=vecs,
               ke_v=kmax,
               c=c,
               r=r)
    return out


def fit(kpts_kv, eps_k, thirdorder=False):
    import numpy.linalg as la
    A_kp = model(kpts_kv)
    if not thirdorder:
        A_kp = A_kp[:, :9]
    return la.lstsq(A_kp, eps_k, recond=-1)



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
                     k_kx*k_ky,
                     k_kx*k_kz,
                     k_ky*k_kz,
                     k_kx,
                     k_ky,
                     k_kz
                     ones,
                     k_kx**3,
                     k_ky**3,
                     k_kz**3,
                     k_kx**2*k_ky,
                     k_kx**2*k_kz,
                     k_ky**2*k_kx,
                     k_ky**2*k_kz,
                     k_kz**2*k_kx,
                     k_kz**2*k_ky]).T
    return A_dp


def _savemass(soc, bt, mass):
    from ase.parallel import world
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




def collect_data(atoms):
    raise NotImplementedError

def webpanel(row, key_descriptions):
    raise NotImplementedError

            
group = 'Property'


if __name__ == '__main__':
    main(standalone_mode=False)
