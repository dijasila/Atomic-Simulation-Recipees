import typing
import numpy as np

from ase.spectrum.band_structure import BandStructure

from asr.core import ASRResult, prepare_result
from asr.result.randomresults import GWInfo
from asr.extra_fluff import _absorption, gaps_from_row
from asr.panels.createwebpanel import (
    PlasmaWebpanel, RamanWebpanel, BSEWebpanel, GwWebpanel, HSEWebpanel
)


@prepare_result
class PlasmaResult(ASRResult):

    plasmafreq_vv: typing.List[typing.List[float]]
    plasmafrequency_x: float
    plasmafrequency_y: float

    key_descriptions = {
        "plasmafreq_vv": "Plasma frequency tensor [Hartree]",
        "plasmafrequency_x": "KVP: 2D plasma frequency (x)"
        "[`eV/Å^0.5`]",
        "plasmafrequency_y": "KVP: 2D plasma frequency (y)"
        "[`eV/Å^0.5`]",
    }
    formats = {"ase_webpanel": PlasmaWebpanel}


@prepare_result
class RamanResult(ASRResult):
    freqs_l: typing.List[float]
    wavelength_w: typing.List[float]
    amplitudes_vvwl: typing.List[
        typing.List[typing.List[typing.List[complex]]]]

    key_descriptions = {
        "freqs_l": "Phonon frequencies (the Gamma point) [1/cm]",
        "wavelength_w": "Laser excitation wavelength [nm]",
        "amplitudes_vvwl": "Raman tensor [a.u.]",
    }
    formats = {"ase_webpanel": RamanWebpanel}

    @staticmethod
    def raman(row, filename):
        # Import the required modules
        import matplotlib.pyplot as plt

        # All required settings
        params = {'broadening': 3.0,  # in cm^-1
                  'wavelength': 532.0,  # in nm
                  'polarization': ['xx', 'yy', 'zz'],
                  'temperature': 300}

        # Read the data from the disk
        data = row.data.get('results-asr.raman.json')

        # If no data, return
        if data is None:
            return

        # Lorentzian function definition
        def lor(w, g):
            lor = 0.5 * g / (np.pi * (w.real**2 + 0.25 * g**2))
            return lor

        from math import pi, sqrt
        # Gaussian function definition

        def gauss(w, g):
            gauss = 1 / (g * sqrt(2 * pi)) * np.exp(-0.5 * w**2 / g**2)
            gauss[gauss < 1e-16] = 0
            return gauss

        # Compute spectrum based on a set of resonances
        from ase.units import kB
        cm = 1 / 8065.544
        kbT = kB * params['temperature'] / cm

        def calcspectrum(wlist, rlist, ww, gamma=10, shift=0, kbT=kbT):
            rr = np.zeros(np.size(ww))
            for wi, ri in zip(wlist, rlist):
                if wi > 1e-1:
                    nw = 1 / (np.exp(wi / kbT) - 1)
                    curr = (1 + nw) * np.abs(ri)**2
                    rr = rr + curr * gauss(ww - wi - shift, gamma)
            return rr

        # Make a latex type formula
        def getformula(matstr):
            matformula = r''
            for ch in matstr:
                if ch.isdigit():
                    matformula += '$_' + ch + '$'
                else:
                    matformula += ch
            return matformula

        # Set the variables and parameters
        wavelength_w = data['wavelength_w']
        freqs_l = data['freqs_l']
        amplitudes_vvwl = data['amplitudes_vvwl']
        selpol = params['polarization']
        gamma = params['broadening']

        # If the wavelength was not found, return
        waveind = int(np.where(wavelength_w == params['wavelength'])[0])
        if not waveind:
            return

        # Check the data to be consistent
        ampshape = amplitudes_vvwl.shape
        freqshape = len(freqs_l)
        waveshape = len(wavelength_w)
        if (ampshape[0] != 3) or (ampshape[1] != 3) or (
            ampshape[2] != waveshape) or (ampshape[3] != freqshape):
            return

        # Make the spectrum
        maxw = min([int(np.max(freqs_l) + 200), int(1.2 * np.max(freqs_l))])
        minw = -maxw / 100
        ww = np.linspace(minw, maxw, 2 * maxw)
        rr = {}
        maxr = np.zeros(len(selpol))
        for ii, pol in enumerate(selpol):
            d_i = 0 * (pol[0] == 'x') + 1 * (pol[0] == 'y') + 2 * (
                    pol[0] == 'z')
            d_o = 0 * (pol[1] == 'x') + 1 * (pol[1] == 'y') + 2 * (
                    pol[1] == 'z')
            rr[pol] = calcspectrum(
                freqs_l, amplitudes_vvwl[d_i, d_o, waveind], ww, gamma=gamma)
            maxr[ii] = np.max(rr[pol])

        # Make the figure panel and add y=0 axis
        ax = plt.figure().add_subplot(111)
        ax.axhline(y=0, color="k")

        # Plot the data and add the axis labels
        for ipol, pol in enumerate(selpol):
            ax.plot(ww, rr[pol] / np.max(maxr), c='C' + str(ipol), label=pol)
        ax.set_xlabel('Raman shift (cm$^{-1}$)')
        ax.set_ylabel('Raman intensity (a.u.)')
        ax.set_ylim((-0.1, 1.1))
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim((minw, maxw))

        # Add the legend to figure
        ax.legend()

        # Count the modes and their degeneracy factors
        w_l, rep_l = RamanResult.count_deg(freqs_l)

        # Add the phonon bars to the figure with showing their degeneracy factors
        pltbar = plt.bar(w_l, -0.04, width=maxw / 100, color='k')
        for idx, rect in enumerate(pltbar):
            ax.text(rect.get_x() + rect.get_width() / 2., -0.1,
                    str(int(rep_l[idx])), ha='center', va='bottom', rotation=0)

        # Remove the extra space and save the figure
        plt.tight_layout()
        plt.savefig(filename)

    @staticmethod
    def count_deg(freqs_l, freq_err=2):

        # Degeneracy factor for modes
        w_l = [freqs_l[0]]
        rep_l = [1]
        # Loop over modes
        for wss in freqs_l[1:]:
            ind = len(w_l) - 1
            if np.abs(w_l[ind] - wss) > freq_err:
                w_l.append(wss)
                rep_l.append(1)
            else:
                rep_l[ind] += 1
        w_l = np.array(w_l)
        rep_l = np.array(rep_l)
        # Return the output
        return w_l, rep_l


@prepare_result
class BSEResult(ASRResult):
    E_B: float
    bse_alphax_w: typing.List[float]
    bse_alphay_w: typing.List[float]
    bse_alphaz_w: typing.List[float]
    key_descriptions = {
        "E_B": ('The exciton binding energy from the Bethe–Salpeter '
                'equation (BSE) [eV].'),
        'bse_alphax_w': 'BSE polarizability x-direction.',
                        'bse_alphay_w': 'BSE polarizability y-direction.',
                        'bse_alphaz_w': 'BSE polarizability z-direction.'}

    formats = {"ase_webpanel": BSEWebpanel}

    @staticmethod
    def absorption(row, filename, direction='x'):
        delta_bse, delta_rpa = gaps_from_row(row)
        return _absorption(
            dim=sum(row.toatoms().pbc),
            magstate=row.magstate,
            gap_dir=row.gap_dir,
            gap_dir_nosoc=row.gap_dir_nosoc,
            bse_data=np.array(
                row.data['results-asr.bse.json'][f'bse_alpha{direction}_w']),
            pol_data=row.data.get('results-asr.polarizability.json'),
            delta_bse=delta_bse,
            delta_rpa=delta_rpa,
            direction=direction,
            filename=filename)


@prepare_result
class GwResult(ASRResult):
    from ase.spectrum.band_structure import BandStructure
    vbm_gw_nosoc: float
    cbm_gw_nosoc: float
    gap_dir_gw_nosoc: float
    gap_gw_nosoc: float
    kvbm_nosoc: typing.List[float]
    kcbm_nosoc: typing.List[float]
    vbm_gw: float
    cbm_gw: float
    gap_dir_gw: float
    gap_gw: float
    kvbm: typing.List[float]
    kcbm: typing.List[float]
    efermi_gw_nosoc: float
    efermi_gw_soc: float
    bandstructure: BandStructure
    key_descriptions = {
        "vbm_gw_nosoc": "Valence band maximum w/o soc. (G₀W₀) [eV]",
        "cbm_gw_nosoc": "Conduction band minimum w/o soc. (G₀W₀) [eV]",
        "gap_dir_gw_nosoc": "Direct gap w/o soc. (G₀W₀) [eV]",
        "gap_gw_nosoc": "Gap w/o soc. (G₀W₀) [eV]",
        "kvbm_nosoc": "k-point of G₀W₀ valence band maximum w/o soc",
        "kcbm_nosoc": "k-point of G₀W₀ conduction band minimum w/o soc",
        "vbm_gw": "Valence band maximum (G₀W₀) [eV]",
        "cbm_gw": "Conduction band minimum (G₀W₀) [eV]",
        "gap_dir_gw": "Direct band gap (G₀W₀) [eV]",
        "gap_gw": "Band gap (G₀W₀) [eV]",
        "kvbm": "k-point of G₀W₀ valence band maximum",
        "kcbm": "k-point of G₀W₀ conduction band minimum",
        "efermi_gw_nosoc": "Fermi level w/o soc. (G₀W₀) [eV]",
        "efermi_gw_soc": "Fermi level (G₀W₀) [eV]",
        "bandstructure": "GW bandstructure."
    }
    formats = {"ase_webpanel": GwWebpanel}

    @staticmethod
    def get_gw_info(row):
        return GWInfo(row)


@prepare_result
class HSEResult(ASRResult):

    vbm_hse_nosoc: float
    cbm_hse_nosoc: float
    gap_dir_hse_nosoc: float
    gap_hse_nosoc: float
    kvbm_nosoc: typing.List[float]
    kcbm_nosoc: typing.List[float]
    vbm_hse: float
    cbm_hse: float
    gap_dir_hse: float
    gap_hse: float
    kvbm: typing.List[float]
    kcbm: typing.List[float]
    efermi_hse_nosoc: float
    efermi_hse_soc: float
    bandstructure: BandStructure

    key_descriptions = {
        "vbm_hse_nosoc": "Valence band maximum w/o soc. (HSE06) [eV]",
        "cbm_hse_nosoc": "Conduction band minimum w/o soc. (HSE06) [eV]",
        "gap_dir_hse_nosoc": "Direct gap w/o soc. (HSE06) [eV]",
        "gap_hse_nosoc": "Band gap w/o soc. (HSE06) [eV]",
        "kvbm_nosoc": "k-point of HSE06 valence band maximum w/o soc",
        "kcbm_nosoc": "k-point of HSE06 conduction band minimum w/o soc",
        "vbm_hse": "KVP: Valence band maximum (HSE06) [eV]",
        "cbm_hse": "KVP: Conduction band minimum (HSE06) [eV]",
        "gap_dir_hse": "KVP: Direct band gap (HSE06) [eV]",
        "gap_hse": "KVP: Band gap (HSE06) [eV]",
        "kvbm": "k-point of HSE06 valence band maximum",
        "kcbm": "k-point of HSE06 conduction band minimum",
        "efermi_hse_nosoc": "Fermi level w/o soc. (HSE06) [eV]",
        "efermi_hse_soc": "Fermi level (HSE06) [eV]",
        "bandstructure": "HSE06 bandstructure."
    }
    formats = {"ase_webpanel": HSEWebpanel}

    @staticmethod
    def get_hse_info(row):
        from asr.result.randomresults import HSEInfo
        return HSEInfo(row)
