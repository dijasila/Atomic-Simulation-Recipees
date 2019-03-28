def absorption(row, fnamein, fnameout):
    def xlim(delta_bse):
        return (0, 5 + delta_bse)

    def ylim(freq, data, delta_bse):
        x1, x2 = xlim(delta_bse)
        i2 = abs(freq - x2).argmin()
        return (0, data[:i2].max() * 1.02)

    def pol2abs(frequencies, pol):
        """absorption in percentage
        """
        x = 4 * np.pi * frequencies * alpha / Ha / Bohr
        return x * pol * 100

    if 'bse_pol' in row.data or 'absorptionspectrum' in row.data:
        if 'bse_pol' in row.data:
            ax = plt.figure().add_subplot(111)
            dir_gap_nosoc = row.get('dir_gap_nosoc')
            dir_gap = row.get('dir_gap')
            if dir_gap is None or dir_gap_nosoc is None:
                delta_bse = 0.0
                delta_rpa = 0.0
                dir_gap_x = None
            else:
                for method in ['_gw', '_hse', '_gllbsc', '']:
                    gapkey = 'dir_gap{}'.format(method)
                    if gapkey in row:
                        dir_gap_x = row.get(gapkey)
                        delta_bse = dir_gap_x - dir_gap
                        delta_rpa = dir_gap_x - dir_gap_nosoc
                        break
            a = row.data.bse_pol
            abs_in = pol2abs(a.freq + delta_bse, a.par)
            ax.plot(a.freq + delta_bse, abs_in, label='BSE', c='k')
            ymax2 = ylim(a.freq + delta_bse, abs_in, delta_bse)[1]
            if 'absorptionspectrum' in row.data:
                freq = row.data.absorptionspectrum.frequencies
                abs_in = pol2abs(freq + delta_rpa,
                                 row.data.absorptionspectrum.alphax_w.imag)
                ax.plot(freq + delta_rpa, abs_in, label='RPA', c='C0')
                ymin, ymax1 = ylim(freq + delta_rpa, abs_in, delta_bse)
                ymax = ymax1 if ymax1 > ymax2 else ymax2
                ax.set_ylim((ymin, ymax))
            if dir_gap_x is not None:
                ax.axvline(dir_gap_x, ls='--', c='0.5', label='Direct QP gap')
            ax.set_title('x-direction')
            ax.set_xlabel('energy [eV]')
            ax.set_ylabel('absorbance [%]')
            ax.legend()
            ax.set_xlim(xlim(delta_bse))
            plt.savefig(fnamein)
            plt.close()
            ax = plt.figure().add_subplot(111)
            if 'bse_pol' in row.data:
                a = row.data.bse_pol
                if len(a.freq) != len(a.per):
                    plt.close()
                    return
                abs_out = pol2abs(a.freq + delta_bse, a.per)
                ax.plot(a.freq + delta_bse, abs_out, label='BSE', c='k')
                ymax2 = ylim(a.freq + delta_bse, abs_out, delta_bse)[1]
            if 'absorptionspectrum' in row.data:
                freq = row.data.absorptionspectrum.frequencies
                abs_out = pol2abs(freq + delta_rpa,
                                  row.data.absorptionspectrum.alphaz_w.imag)
                ax.plot(freq + delta_rpa, abs_out, label='RPA', c='C0')
                ymin, ymax1 = ylim(freq + delta_rpa, abs_out, delta_bse)
                ymax = ymax1 if ymax1 > ymax2 else ymax2
                ax.set_ylim((ymin, ymax))
            if dir_gap_x is not None:
                ax.axvline(dir_gap_x, ls='--', c='0.5', label='Direct QP gap')
            ax.set_title('z-direction')
            ax.set_xlabel('energy [eV]')
            ax.set_ylabel('absorbance [%]')
            ax.legend()
            ax.set_xlim(xlim(delta_bse))
            plt.tight_layout()
            plt.savefig(fnameout)
            plt.close()


# def webpanel(row, key_descriptions):
#     from asr.custom import fig, table
#     # only show bse if binding energy is there
#     if row.get('bse_binding', 0) > 0:
#         bse_binding = table('Property',
#                             ['bse_binding', 'excitonmass1', 'excitonmass2'],
#                             key_descriptions)
#     else:
#         bse_binding = table('Property', [])

#     panel = ('Optical absorption (BSE)', [[fig('abs-in.png'), bse_binding],
#                                           [fig('abs-out.png')]])

#     things = [(absorption, ['abs-in.png', 'abs-out.png'])]

#     return panel, things


group = 'Property'
