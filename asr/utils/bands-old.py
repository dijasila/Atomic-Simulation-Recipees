def bs_from_gpw(fname, soc, evac):
    from gpaw import GPAW
    import numpy as np
    calc = GPAW(fname, txt=None)
    bs = calc.band_structure()
    x, X, labels = bs.get_labels()
    if soc:
        from gpaw.spinorbit import soc_eigenstates
        spin = soc_eigenstates(calc)
        e = spin.eigenvalues().T
        ef = spin.fermi_level
    else:
        e = bs.energies
        ef = bs.reference
    if evac:
        evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
    else: 
        print('Setting Evac to 0.0. Specify evac=True if you want to explicitly calculate it.')
        evac = 0.0
    return dict(energies=e,
                efermi=ef,
                evac=evac,
                kpts=x,
                specpts=X,
                labels=labels)


def bs_from_json(fname):
    from ase.io.jsonio import read_json
    bs = read_json(self.filename)
    try:
        path = bs['path']
        e = bs['energies']
        ef = bs['reference']
    except TypeError:
        path = bs.path
        e = bs.energies
        ef = bs.reference
    try:
        evac = bs.evac
    except AttributeError:
        print('No vacuum level found. Evac set to 0.0')
        evac = 0.0
    x, X, labels = path.get_linear_kpoint_axis()
    return dict(energies=e,
                efermi=ef,
                evac=evac,
                kpts=x,
                specpts=X,
                labels=labels)


class Bands:
    def __init__(self, 
                 filename, 
                 soc=False, 
                 evac=False, 
                 gw=False, 
                 gs='results-asr.gs.json'):

        from os.path import splitext

        self.filename = filename
        self._basename, self._ext = splitext(filename)

        print(f'Reading stuff from file {filename}...')

        if self._ext == '.gpw':
            #data = bs_from_gpw(self.filename, soc, evac)

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
                e = bs.energies
                ef = bs.reference
            if evac:
                evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
            else: 
                evac = 0.0

        if self._ext == '.json' and not gw:
            from ase.io.jsonio import read_json

            bs = read_json(self.filename)

            try:
                path = bs['path']
                e = bs['energies']
                ef = bs['reference']
            except TypeError:
                path = bs.path
                e = bs.energies
                ef = bs.reference

            try:
                evac = bs.evac
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
        self.bs = bs

        print("Done")

    def plot(self, 
             reference="evac", 
             style=None, 
             ylim=None, 
             xtitle=None,
             ytitle=None,
             title=None,
             label=None,
             show=False,
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

        energies = self.energies.T - ref
        efermi = self.efermi - ref

        if not ylim:
            ylim = [efermi-4, efermi+4]
        if not style:
            style = dict(
                color='C1',
                ls='-',
                lw=1.5)

        ax = plt.figure(figsize=(12, 9)).add_subplot(111)
        
        ax.plot(self.kpts, energies[0], **style, label=label)
        for band in energies[1:]:
            ax.plot(self.kpts, band, **style)
        for spec in self.specpts[1:-1]:
            ax.axvline(spec, color='#bbbbbb')
        ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in self.labels], fontsize=fontsize2)

        if not hlines:
            ax.axhline(efermi, color='#bbbbbb', ls='--', lw=1.5)
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
        if show is True:
            plt.show()

    def dump_to_json(self, filename=None):
        from ase.io.jsonio import write_json
        if not filename:
            filename = f'{self._basename}.json'
            write_json(filename, self.bs)

    def get_gap_and_edges(self, reference='evac'):
        en = self.energies.T
        allcb = en[(en - self.efermi) > 0]
        allvb = en[(en - self.efermi) < 0]
        cbm = allcb.min()
        vbm = allvb.max()
        print('Returning, in order: cbm, vbm, band gap')
        return cbm, vbm, cbm - vbm








