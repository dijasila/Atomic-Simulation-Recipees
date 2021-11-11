import numpy as np
from typing import Union
from gpaw import GPAW


def bs_from_gpw(fname):
    from gpaw import GPAW
    calc = GPAW(fname, txt=None)
    return calc.band_structure(), calc.get_fermi_level()


def bs_from_json(fname, soc):
    from ase.io.jsonio import read_json
    from ase.spectrum.band_structure import BandStructure
    dct = read_json(fname)
    if isinstance(dct, BandStructure):
        return dct
    if soc:
        return BandStructure(dct['path'],
                             [dct['eigenvalues_soc']],
                             dct['evac']), \
               dct['efermi_soc'], \
               dct
    else:
        return BandStructure(dct['path'],
                             [dct['eigenvalues_nosoc']],
                             dct['evac']), \
               dct['efermi_nosoc'], \
               dct


def calculate_evac(calc):
    '''Obtain vacuum level from a GPAW calculator'''
    evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
    return evac


def get_cb_vb_surface(calc: Union[GPAW, dict], all_bz: bool = True, soc: bool = False):
    '''Returns the surface formed by the edge states of CB and VB
       calculated on a 2D k-point grid
    '''
    if isinstance(calc, GPAW):
        ef = calc.get_fermi_level()
        if all_bz:
            bz = calc.get_bz_k_points()
            bzmap = calc.get_bz_to_ibz_map()
            evs = np.array([calc.get_eigenvalues(kpt=i) for i in bzmap])
        else:
            bz = calc.get_ibz_k_points()
            evs = np.array([calc.get_eigenvalues(kpt=i) for i in range(len(bz))])

    elif isinstance(calc, dict):
        if soc is True:
            ef = calc['efermi_soc']
            evs = calc['eigenvalues_soc']
        else:
            ef = calc['efermi_nosoc']
            evs = calc['eigenvalues_nosoc']
        bz = np.asarray(calc['kpts'])

    cb = []
    vb = []
    for ev in evs:
        cbi = ev[ev - ef > 0]
        vbi = ev[ev - ef < 0]
        cb.append(cbi.min())
        vb.append(vbi.max())

    return bz[:, 0:2], np.asarray(cb), np.asarray(vb)


def plot_cb_vb_surface(calc, all_bz: bool = True, title: str = '', soc: bool = False):
    '''shows the brillouin zone in a 3D plot'''
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    bz, cb, vb = get_cb_vb_surface(calc, all_bz=all_bz, soc=soc)
    vbm_pos = np.argmax(vb)
    cbm_pos = np.argmin(cb)
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_trisurf(bz[:,0], bz[:,1], cb, cmap='winter')
    ax.plot_trisurf(bz[:,0], bz[:,1], vb, cmap='summer')
    ax.plot(bz[vbm_pos, 0], bz[vbm_pos, 1], vb[vbm_pos], marker='o', zorder=3)
    ax.plot(bz[cbm_pos, 0], bz[cbm_pos, 1], cb[vbm_pos], marker='o', zorder=4)
    ax.set_title(title)
    ax.set_xlabel('$k_x$', labelpad=20)
    ax.set_ylabel('$k_y$', labelpad=20)
    ax.set_zlabel('Energy (eV)', labelpad=20)
    plt.show()


def get_bz_eigenvalues(calc):
    import numpy as np
    bzmap = calc.get_bz_to_ibz_map()
    return np.array([calc.get_eigenvalues(kpt=i) for i in bzmap])


def direct_gap(calc):
    '''Obtain lowest direct transition on a 2D k-point grid'''
    cb, vb = get_cb_vb_surface(calc)
    dir_gaps = cb - vb
    return dir_gaps.min()


def is_gap_direct(calc):
    '''Return True if the gap is direct'''
    ibz = calc.get_ibz_k_points()
    cb, vb = get_cb_vb_surface(calc)
    cbm_k = ibz[np.argmin(cb)]
    vbm_k = ibz[np.argmax(vb)]
    return np.allclose(cbm_k, vbm_k)


def multiplot(*toplot,
              reference: str = 'evac',
              ylim=None,
              title=None,
              xtitle=None,
              ytitle=None,
              labels=None,
              hlines=None,
              styles=None,
              colors=None,
              fermiline=True,
              customticks=None,
              show=True,
              legend=True,
              fontsize1=24,
              fontsize2=22,
              loc='best'):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import warnings
    warnings.filterwarnings('ignore')

    if not colors:
        colors = [cm.Set2(i) for i in range(len(toplot))]

    ax = plt.figure(figsize=(12, 9)).add_subplot(111)

    items = []
    for tp in toplot:
        if isinstance(tp, str):
            items.append(JsonBands(tp))
        elif isinstance(tp, JsonBands):
            items.append(tp)
        else:
            msg = 'Please provide either Bands objects or filenames'
            raise ValueError(msg)

    kpts, specpts, lbls = items[0].get_kpts_and_labels(normalize=True)
    if customticks:
        lbls = list(customticks)
    ticks = [lab.replace('G', r'$\Gamma$') for lab in lbls]
    ax.set_xticklabels(ticks, fontsize=fontsize2)
    ax.set_xticks(specpts)
    ax.set_xlim([kpts[0], kpts[-1]])

    allfermi = []

    for (index, item), color in zip(enumerate(items), colors):
        if reference == "evac":
            ref = item.reference
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{vac}}$ (eV)'
        elif reference == "ef":
            '''Useful only for plotting single bandstructure'''
            ref = item.get_efermi()
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{F}}$ (eV)'
        elif isinstance(reference, float):
            ref = reference

        energies = item.get_energies() - ref
        efermi = item.get_efermi() - ref
        allfermi.append(efermi)

        try:
            lbl = labels[index]
        except (TypeError, IndexError):
            lbl = item._basename
        try:
            style = styles[index]
        except (TypeError, IndexError):
            color = colors[index]
            style = dict(ls='-', color=color, lw=2.0)

        kpts, _, _ = item.get_kpts_and_labels(normalize=True)
        ax.plot(kpts, energies[0], **style, label=lbl)
        for band in energies[1:]:
            ax.plot(kpts, band, **style)
        if fermiline:
            ax.axhline(efermi, color='k', ls='--', lw=2.0)

    for spec in specpts[1:-1]:
        ax.axvline(spec, color='k', lw=2.0)

    if hlines:
        for val in hlines:
            ax.axhline(val, color='#bbbbbb', ls='--', lw=2.0)

    if title:
        plt.title(title, pad=10, fontsize=fontsize1)
    if not ylim:
        ylim = [min(allfermi) - 4, max(allfermi) + 4]

    ax.set_ylim(ylim)
    ax.set_xlabel(xtitle, fontsize=fontsize1, labelpad=8)
    ax.set_ylabel(ytitle, fontsize=fontsize1, labelpad=8)
    plt.yticks(fontsize=fontsize2)
    ax.xaxis.set_tick_params(width=3, length=10)
    ax.yaxis.set_tick_params(width=3, length=10)
    plt.setp(ax.spines.values(), linewidth=3)

    if legend:
        plt.legend(loc=loc, fontsize=fontsize2 - 2)

    if show:
        plt.show()


class JsonBands:
    def __init__(self,
                 filename,
                 soc: bool = True):

        from os.path import splitext
        self.filename = filename
        self._basename, self._ext = splitext(filename)
        if self._ext != '.json':
            raise ValueError('Not a json file!')

        print(f'Reading stuff from file {filename}...')
        self.bandstructure, self.efermi, self.data = bs_from_json(filename, soc)
        print("Done")

    def get_energies(self):
        return self.bandstructure.energies.T

    def get_efermi(self):
        return self.efermi

    def get_kpts_and_labels(self, normalize=False):
        x, X, lbls = self.bandstructure.get_labels()
        if normalize:
            delta = x[-1] - x[0]
            return (x / delta, X / delta, lbls)
        else:
            return (x, X, lbls)

    def get_evac(self):
        errmsg = 'A .gpw file is required!'
        assert (self._ext == '.gpw'), errmsg
        evac = calculate_evac(self.filename)
        self.evac = evac
        return evac

    def calculate_SOC(self):
        from gpaw.spinorbit import soc_eigenstates
        from gpaw import GPAW
        errmsg = 'A .gpw file is required!'
        assert (self._ext == '.gpw'), errmsg
        calc = GPAW(self.filename, txt='-')
        spin = soc_eigenstates(calc)
        esoc = spin.eigenvalues().T
        efsoc = spin.fermi_level
        self.bandstructure.energies_soc = esoc
        self.bandstructure.reference_soc = efsoc
        return esoc, efsoc

    def plot(self, *args, **kwargs):
        multiplot(self, *args, **kwargs)

    def dump_to_json(self, filename=None):
        from ase.io.jsonio import write_json
        if not filename:
            filename = f'{self._basename}.json'
        dct = self.bandstructure.__dict__.copy()
        try:
            dct['path'] = dct.pop('_path')
            dct['energies'] = dct.pop('_energies')
            dct['reference'] = dct.pop('_reference')
        except KeyError:
            pass
        write_json(filename, dct)

    def get_band_edges(self, reference=0.0):
        en = self.get_energies()
        ef = self.get_efermi()
        allcb = en[(en - ef) > 0]
        allvb = en[(en - ef) < 0]
        cbm = allcb.min()
        vbm = allvb.max()
        if reference == 'evac':
            try:
                ref = self.bandstructure.evac
            except AttributeError:
                print('you have to calculate the vacuum level first!')
                return 0
        elif reference == 'ef':
            ref = self.bandstructure.reference
        elif isinstance(reference, float):
            ref = reference
        return vbm - ref, cbm - ref

    def get_band_edges_at_kpt(self, kpt):
        import numpy as np
        en_T = self.get_energies().T[0]
        ef = self.get_efermi()
        x, X, labels = self.get_kpts_and_labels()
        spec = X[labels.index(kpt)]
        indx_spec = np.where(x == spec)[0][0]
        en_kpt = en_T[indx_spec]
        en_vb = en_kpt[(en_kpt - ef) < 0]
        en_cb = en_kpt[(en_kpt - ef) > 0]
        return en_vb.max(), en_cb.min()

    def get_homo_lumo_gap(self, reference=0.0):
        vbm, cbm = self.get_band_edges(reference=reference)
        return cbm - vbm

    def get_direct_gap(self):
        en_T = self.get_energies().T[0] - self.get_efermi()
        dyn_gap = np.zeros(en_T.shape[0])
        for i, en in enumerate(en_T):
            lumo = en[en > 0].min()
            homo = en[en < 0].max()
            dyn_gap[i] = lumo - homo
        return dyn_gap.min()

    def get_transition(self, kpt1, kpt2):
        edges1 = self.get_band_edges_at_kpt(kpt1)
        edges2 = self.get_band_edges_at_kpt(kpt2)
        return max(edges2) - min(edges1)
