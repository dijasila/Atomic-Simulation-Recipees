def bzcut_pbe(row, pathcb, pathvb, figsize=(6.4, 2.8)):
    from ase.units import Bohr, Ha
    from c2db.em import evalmodel
    from ase.dft.kpoints import kpoint_convert
    from matplotlib import pyplot as plt
    import numpy as np
    labels_from_kpts = None  # XXX: Fix pep8
    sortcolors = True
    erange = 0.05  # energy window
    cb = row.get('data', {}).get('effectivemass', {}).get('cb', {})
    vb = row.get('data', {}).get('effectivemass', {}).get('vb', {})

    def getitsorted(keys, bt):
        keys = [k for k in keys if 'spin' in k and 'band' in k]
        return sorted(
            keys, key=lambda x: int(x.split('_')[1][4:]), reverse=bt == 'vb')

    def get_xkrange(row, erange):
        xkrange = 0.0
        for bt in ['cb', 'vb']:
            xb = row.data.get('effectivemass', {}).get(bt)
            if xb is None:
                continue
            xb0 = xb[getitsorted(xb.keys(), bt)[0]]
            mass_u = abs(xb0['mass_u'])
            xkr = max((2 * mass_u * erange / Ha)**0.5 / Bohr)
            xkrange = max(xkrange, xkr)
        return xkrange

    for bt, xb, path in [('cb', cb, pathcb), ('vb', vb, pathvb)]:
        b_u = xb.get('bzcut_u')
        if b_u is None or b_u == []:
            continue

        xb0 = xb[getitsorted(xb.keys(), bt)[0]]
        mass_u = xb0['mass_u']
        coeff = xb0['c']
        ke_v = xb0['ke_v']
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=figsize,
            sharey=True,
            gridspec_kw={'width_ratios': [1, 1.25]})
        things = None
        xkrange = get_xkrange(row, erange)
        for u, b in enumerate(b_u):  # loop over directions
            ut = u if bt == 'cb' else abs(u - 1)
            ax = axes[ut]
            e_mk = b['e_dft_km'].T - row.evac
            sz_mk = b['sz_dft_km'].T
            if row.get('has_invsymm', 0) == 1:
                sz_mk[:] = 0.0
            kpts_kc = b['kpts_kc']
            xk, _, _ = labels_from_kpts(kpts=kpts_kc, cell=row.cell)
            xk -= xk[-1] / 2
            # fitted model
            xkmodel = xk.copy()  # xk will be permutated
            kpts_kv = kpoint_convert(row.cell, skpts_kc=kpts_kc)
            kpts_kv *= Bohr
            emodel_k = evalmodel(kpts_kv=kpts_kv, c_p=coeff) * Ha
            emodel_k -= row.evac
            # effective mass fit
            emodel2_k = (xkmodel * Bohr)**2 / (2 * mass_u[u]) * Ha
            ecbm = evalmodel(ke_v, coeff) * Ha
            emodel2_k = emodel2_k + ecbm - row.evac
            # dft plot
            shape = e_mk.shape
            x_mk = np.vstack([xk] * shape[0])
            if sortcolors:
                shape = e_mk.shape
                perm = (-sz_mk).argsort(axis=None)
                e_mk = e_mk.ravel()[perm].reshape(shape)
                sz_mk = sz_mk.ravel()[perm].reshape(shape)
                x_mk = x_mk.ravel()[perm].reshape(shape)
            for i, (e_k, sz_k, x_k) in enumerate(zip(e_mk, sz_mk, x_mk)):
                things = ax.scatter(x_k, e_k, c=sz_k, vmin=-1, vmax=1)
            ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
            # ax.plot(xkmodel, emodel_k, c='b', ls='-', label='3rd order')
            sign = np.sign(mass_u[u])
            if (bt == 'cb' and sign > 0) or (bt == 'vb' and sign < 0):
                ax.plot(xkmodel, emodel2_k, c='r', ls='--')
            ax.set_title('Mass {}, direction {}'.format(bt.upper(), ut + 1))
            if bt == 'vb':
                y1 = ecbm - row.evac - erange * 0.75
                y2 = ecbm - row.evac + erange * 0.25
            elif bt == 'cb':
                y1 = ecbm - row.evac - erange * 0.25
                y2 = ecbm - row.evac + erange * 0.75

            ax.set_ylim(y1, y2)
            ax.set_xlim(-xkrange, xkrange)
            ax.set_xlabel(r'$\Delta k$ [1/$\mathrm{\AA}$]')
        if things is not None:
            cbar = fig.colorbar(things, ax=axes[1])
            cbar.set_label(r'$\langle S_z \rangle$')
            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
            cbar.update_ticks()
        fig.tight_layout()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def bz_soc(row, fname):
    from ase.geometry.cell import Cell
    from matplotlib import pyplot as plt
    cell = Cell(row.cell, pbc=row.pbc)
    lat, _ = cell.bravais()
    lat.plot_bz()
    plt.savefig(fname)


def webpanel(row, key_descriptions):
    # XXX we still need to make the zoomed in bs plots
    # things = [(bzcut_pbe, ['pbe-bzcut-cb-bs.png', 'pbe-bzcut-vb-bs.png']),
    #           (bz_soc, ['bz.png'])]
    things = [(bz_soc, ['bz.png'])]

    panel = ()
    return panel, things


group = 'Property'
