class Bands:
    def __init__(self, 
                 filename, 
                 soc=False, 
                 evac=False, 
                 gw=False, 
                 gs='results-asr.gs.json'):

        from os.path import splitext

        self.filename = filename
        self.basename, self.ext = splitext(filename)

        print(f'Reading stuff from file {filename}...')

        if self.ext == '.gpw':
            from gpaw import GPAW
            import numpy as np

            calc = GPAW(self.filename, txt=None)
            bs = calc.band_structure()
            x, X, labels = bs.get_labels()
            if soc:
                from gpaw.spinorbit import soc_eigenstates
                spin = soc_eigenstates(calc)
                e = spin.eigenvalues().T
                ef = spin.fermi_level
            else:
                e = bs.energies.T
                ef = bs.reference
            if evac:
                evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
            else: 
                evac = 0.0

        if self.ext == '.json' and not gw:
            from ase.io.jsonio import read_json

            dct = read_json(self.filename)

            try:
                path = dct['path']
                e = dct['energies']
                ef = dct['reference']
            except TypeError:
                path = dct.path
                e = dct.energies
                ef = dct.reference

            try:
                evac = dct.evac
            except AttributeError:
                evac = 0.0
            x, X, labels = path.get_linear_kpoint_axis()

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

        self.energies = e
        self.efermi = ef
        self.evac = evac
        self.kpts = x
        self.specpts = X
        self.labels = labels

        print("Done")

    def plot(self, 
             reference="evac", 
             style=None, 
             ylim=None, 
             xtitle=None,
             ytitle=None,
             title=None,
             label=None,
             fontsize1=24,
             fontsize2=20,
             vlines=[],
             hlines=[]):

        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore')

        print('Preparing plot...')
        
        if reference == "evac":
            ref = self.evac
            if not ytitle:
                ytitle = '$\mathrm{E-E_{vac}}$'
        elif reference == "ef":
            ref = self.efermi
            if not ytitle:
                ytitle = '$\mathrm{E-E_{F}}$'
        elif isinstance(reference, float):
            ref = reference

        self.energies -= ref
        self.efermi -= ref

        if not ylim:
            ylim = [self.efermi-4, self.efermi+4]
        if not style:
            style = dict(
                color='C1',
                ls='-',
                lw=1.5)

        ax = plt.figure(figsize=(12, 9)).add_subplot(111)
        
        ax.plot(self.kpts, self.energies[0], **style, label=label)
        for band in self.energies[1:]:
            ax.plot(self.kpts, band, **style)
        for spec in self.specpts[1:-1]:
            ax.axvline(spec, color='#bbbbbb')
        ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in self.labels], fontsize=fontsize2)

        if not hlines:
            ax.axhline(self.efermi, color='#bbbbbb', ls='--', lw=1.5)
        else:
            for val in hlines:
                ax.axhline(val, color='#bbbbbb', ls='--', lw=1.5)
        
        if title:
            ax.set_title(title, fontsize = fontsize1)

        ax.set_ylim(ylim)
        ax.set_xlim([self.kpts[0], self.kpts[-1]])
        ax.set_ylabel(ytitle, fontsize=fontsize1, labelpad=8)
        ax.set_xlabel(xtitle, fontsize=fontsize1, labelpad=8)
        ax.set_xticks(self.specpts)
        plt.yticks(fontsize=fontsize2)
        ax.xaxis.set_tick_params(width=3, length=10)
        ax.yaxis.set_tick_params(width=3, length=10)
        plt.setp(ax.spines.values(), linewidth=3)
        if label:
            plt.legend(loc="upper left", fontsize = fontsize2)
        
        print('Done')
        plt.show()

    def dump(self, filename=None):
        if not filename:
            filename = f'{self.basename}.json'



