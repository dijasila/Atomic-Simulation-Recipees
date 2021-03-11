import numpy as np


def bs_from_gpw(fname):
    from gpaw import GPAW
    calc = GPAW(fname, txt=None)
    bs = calc.band_structure()
    return bs


def bs_from_json(fname):
    from ase.io.jsonio import read_json
    from ase.spectrum.band_structure import BandStructure
    read = read_json(fname)
    if isinstance(read, BandStructure):
        return read
    bs = BandStructure(read['path'],
                       read['energies'],
                       read['reference'])
    return bs


def calculate_evac(filename):
    from gpaw import GPAW
    import numpy as np
    calc = GPAW(filename, txt='-')
    evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
    return evac


def multiplot(*toplot,
              reference=None,
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
              fontsize1=24,
              fontsize2=22,
              loc='upper left'):
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
            items.append(Bands(tp))
        elif isinstance(tp, Bands):
            items.append(tp)
        else:
            msg = 'Please provide either Bands objects or filenames'
            raise ValueError(msg)

    kpts, specpts, lbls = items[0].get_kpts_and_labels(normalize=True)
    for spec in specpts[1:-1]:
        ax.axvline(spec, color='#bbbbbb')
    if customticks:
        lbls = list(customticks)
    ticks = [lab.replace('G', r'$\Gamma$') for lab in lbls]
    ax.set_xticklabels(ticks, fontsize=fontsize2)
    ax.set_xticks(specpts)
    ax.set_xlim([kpts[0], kpts[-1]])

    allfermi = []

    for (index, item), color in zip(enumerate(items), colors):
        if reference == "evac":
            ref = item.calculate_evac()
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{vac}}$'
        elif reference == "ef":
            '''Useful only for plotting single bandstructure'''
            ref = item.get_efermi()
        elif isinstance(reference, float):
            ref = reference
        else:
            ref = 0.0

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
            style = dict(ls='-', color=color, lw=1.5)

        kpts, _, _ = item.get_kpts_and_labels(normalize=True)
        ax.plot(kpts, energies[0], **style, label=lbl)
        for band in energies[1:]:
            ax.plot(kpts, band, **style)
        if fermiline:
            ax.axhline(efermi, color='#bbbbbb', ls='--', lw=1.5)

    if hlines:
        for val in hlines:
            ax.axhline(val, color='#bbbbbb', ls='--', lw=1.5)

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

    plt.legend(loc=loc, fontsize=fontsize2 - 2)

    if show:
        plt.show()


class Bands:
    def __init__(self,
                 filename,
                 gw=False,
                 gs='results-asr.gs.json'):

        from os.path import splitext
        self.filename = filename
        self._basename, self._ext = splitext(filename)

        print(f'Reading stuff from file {filename}...')

        if self._ext == '.gpw':
            self.bandstructure = bs_from_gpw(self.filename)

        if self._ext == '.json' and not gw:
            self.bandstructure = bs_from_json(self.filename)

        '''
        if gw:
            gsdct = read_json(gs)
            gsdata = gsdct["kwargs"]["data"]
            evac = gsdata["evac"]
            dct = read_json(self.filename)
            data = dct["kwargs"]["data"]
            path = data['bandstructure']['path']
            ef = data['efermi_gw_soc'] - evac1
            e1 = data1['bandstructure']['e_int_mk'] - evac1
            vbm1 = data1["vbm_gw"] - evac1
            cbm1 = data1["cbm_gw"] - evac1
            edg1 = get_edges(e1, ef1)
            x1, X1, labels1 = path1.get_linear_kpoint_axis()
        '''

        print("Done")

    def get_energies(self):
        return self.bandstructure.energies.T

    def get_efermi(self):
        return self.bandstructure.reference

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
