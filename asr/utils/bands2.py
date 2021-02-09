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


def multiplot(*files,
              reference=None,
              ylim=None,
              title=None,
              xtitle=None,
              ytitle=None,
              labels=None,
              hlines=None,
              styles=None,
              show=True,
              fontsize1=24,
              fontsize2=20):
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    ax = plt.figure(figsize=(12, 9)).add_subplot(111)

    items = []
    for f in files:
        items.append(Bands(f))

    kpts, specpts, labels = items[0].get_kpts_and_labels()
    for spec in specpts[1:-1]:
        ax.axvline(spec, color='#bbbbbb')
    ticks = [lab.replace('G', r'$\Gamma$') for lab in labels]
    ax.set_xticklabels(ticks, fontsize=fontsize2)
    ax.set_xticks(specpts)
    ax.set_xlim([kpts[0], kpts[-1]])

    allfermi = []

    for index, item in enumerate(items):
        if reference == "evac":
            ref = item.calculate_evac()
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{vac}}$'
        elif reference == "ef":
            '''Useful only for plotting single bandstructure'''
            item.get_efermi()
        elif isinstance(reference, float):
            ref = reference
        else:
            ref = 0.0
        energies = item.get_energies() - ref
        efermi = item.get_efermi() - ref
        allfermi.append(efermi)
        try:
            lbl = labels[index]
            style = styles[index]
        except (TypeError, IndexError) as errors:
            lbl = item._basename
            style = dict(ls='-', lw=1.5)
        ax.plot(kpts, energies[0], **style, label=lbl)
        for band in energies[1:]:
            ax.plot(kpts, band, **style)

    if hlines:
        for val in hlines:
            ax.axhline(val, color='#bbbbbb', ls='--', lw=1.5)

    if title:
        plt.title(title, pad=10)
    if not ylim:
        ylim = [min(allfermi) - 4, max(allfermi) + 4]
    ax.set_ylim(ylim)
    ax.set_xlabel(xtitle, fontsize=fontsize1, labelpad=8)
    ax.set_ylabel(ytitle, fontsize=fontsize1, labelpad=8)
    plt.yticks(fontsize=fontsize2)
    ax.xaxis.set_tick_params(width=3, length=10)
    ax.yaxis.set_tick_params(width=3, length=10)
    plt.setp(ax.spines.values(), linewidth=3)

    plt.legend(loc="upper left", fontsize=fontsize2)

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

        print("Done")

    def get_energies(self):
        return self.bandstructure.energies.T

    def get_efermi(self):
        return self.bandstructure.reference

    def get_kpts_and_labels(self):
        return self.bandstructure.get_labels()

    def get_ref_for_plot(self, label):
        if label == "evac":
            try:
                ref = self.bandstructure.evac
            except AttributeError:
                ref = self.calculate_evac()
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{vac}}$'
        elif label == "ef":
            ref = self.get_efermi()
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{F}}$'
        elif isinstance(label, float):
            ref = label

    def calculate_evac(self):
        from gpaw import GPAW
        import numpy as np
        errmsg = 'A .gpw file is required!'
        assert (self._ext == '.gpw'), errmsg
        calc = GPAW(self.filename, txt=None)
        evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
        self.bandstructure.evac = evac
        return evac

    def calculate_SOC(self):
        from gpaw.spinorbit import soc_eigenstates
        from gpaw import GPAW
        errmsg = 'A .gpw file is required!'
        assert (self._ext == '.gpw'), errmsg
        calc = GPAW(self.filename, txt=None)
        spin = soc_eigenstates(calc)
        esoc = spin.eigenvalues().T
        efsoc = spin.fermi_level
        self.bandstructure.energies_soc = esoc
        self.bandstructure.reference_soc = efsoc
        return esoc, efsoc

    def plot(self,
             reference=0.0,
             style=None,
             ylim=None,
             xtitle=None,
             ytitle=None,
             title=None,
             label=None,
             show=True,
             fontsize1=24,
             fontsize2=20,
             vlines=[],
             hlines=[]):

        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore')

        print('Preparing plot...')

        if reference == "evac":
            try:
                ref = self.bandstructure.evac
            except AttributeError:
                ref = self.calculate_evac()
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{vac}}$'
        elif reference == "ef":
            ref = self.bandstructure.reference
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{F}}$'
        elif isinstance(reference, float):
            ref = reference

        energies = self.get_energies() - ref
        efermi = self.get_efermi() - ref

        if not ylim:
            ylim = [efermi - 4, efermi + 4]
        if not style:
            style = dict(
                color='C1',
                ls='-',
                lw=1.5)

        ax = plt.figure(figsize=(12, 9)).add_subplot(111)
        kpts, specpts, labels = self.get_kpts_and_labels()
        if label:
            lbl = label
        else:
            lbl = self._basename

        ax.plot(kpts, energies[0], **style, label=lbl)
        for band in energies[1:]:
            ax.plot(kpts, band, **style)
        for spec in specpts[1:-1]:
            ax.axvline(spec, color='#bbbbbb')
        ax.set_xticklabels([lab.replace('G', r'$\Gamma$')
                            for lab in labels], fontsize=fontsize2)

        if not hlines:
            ax.axhline(efermi, color='#bbbbbb', ls='--', lw=1.5)
        else:
            for val in hlines:
                ax.axhline(val, color='#bbbbbb', ls='--', lw=1.5)

        if title:
            ax.set_title(title, fontsize=fontsize1)

        ax.set_xlim([kpts[0], kpts[-1]])
        ax.set_ylim(ylim)
        ax.set_xlabel(xtitle, fontsize=fontsize1, labelpad=8)
        ax.set_ylabel(ytitle, fontsize=fontsize1, labelpad=8)
        ax.set_xticks(specpts)
        plt.yticks(fontsize=fontsize2)
        ax.xaxis.set_tick_params(width=3, length=10)
        ax.yaxis.set_tick_params(width=3, length=10)
        plt.setp(ax.spines.values(), linewidth=3)

        if label:
            plt.legend(loc="upper left", fontsize=fontsize2)
        print('Done')

        if show:
            plt.show()

    def dump_to_json(self, filename=None):
        from ase.io.jsonio import write_json
        if not filename:
            filename = f'{self._basename}.json'
            write_json(filename, self.bandstructure.__dict__)

    def get_gap_and_edges(self, reference=0.0):
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
                ref = self.calculate_evac()
        elif reference == 'ef':
            ref = self.bandstructure.reference
        elif isinstance(reference, float):
            ref = reference
        return cbm - ref, vbm - ref, cbm - vbm
