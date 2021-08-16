"""Raman spectrum."""
import typing
import numpy as np
from asr.core import command, ASRResult, prepare_result, atomsopt
from asr.database.browser import fig, make_panel_description, href, describe_entry

from ase import Atoms


panel_description = make_panel_description(
    """Raman spectroscopy relies on inelastic scattering of photons by optical
phonons. The Stokes part of the Raman spectrum, corresponding to emission of a
single Gamma-point phonon is calculated for different incoming/outgoing photon
polarizations using third order perturbation theory.""",
    articles=[
        href("""A. Taghizadeh et al.  A library of ab initio Raman spectra for automated
identification of 2D materials. Nat Commun 11, 3011 (2020).""",
             'https://doi.org/10.1038/s41467-020-16529-6'),
    ],
)


def webpanel(result, context):

    # Make a table from the phonon modes
    data = result
    if data:
        table = []
        freqs_l = data['freqs_l']
        w_l, rep_l = count_deg(freqs_l)
        # print(w_l)
        # print(rep_l)
        nph = len(w_l)
        for ii in range(nph):
            key = 'Mode {}'.format(ii + 1)
            table.append(
                (key,
                 np.array2string(
                     np.abs(
                         w_l[ii]),
                     precision=1),
                    rep_l[ii]))
        opt = {'type': 'table',
               'header': ['Mode', 'Frequency (1/cm)', 'Degeneracy'],
               'rows': table}
    else:
        opt = None
    # Make the panel
    panel = {'title': describe_entry('Raman spectrum', panel_description),
             'columns': [[fig('Raman.png')], [opt]],
             'plot_descriptions':
                 [{'function': raman,
                   'filenames': ['Raman.png']}],
             'sort': 22}

    return [panel]


@prepare_result
class Result(ASRResult):

    freqs_l: typing.List[float]
    wavelength_w: typing.List[float]
    amplitudes_vvwl: typing.List[typing.List[typing.List[typing.List[complex]]]]

    key_descriptions = {
        "freqs_l": "Phonon frequencies (the Gamma point) [1/cm]",
        "wavelength_w": "Laser excitation wavelength [nm]",
        "amplitudes_vvwl": "Raman tensor [a.u.]",
    }
    formats = {'webpanel2': webpanel}


@command('asr.c2db.raman')
@atomsopt
def main(atoms: Atoms) -> Result:
    raise NotImplementedError


def raman(context, filename):
    # Import the required modules
    import matplotlib.pyplot as plt

    # All required settings
    params = {'broadening': 3.0,  # in cm^-1
              'wavelength': 532.0,  # in nm
              'polarization': ['xx', 'yy', 'zz'],
              'temperature': 300}

    # Read the data from the disk
    data = context.result

    # Lorentzian function definition
    def lor(w, g):
        lor = 0.5 * g / (np.pi * ((w.real)**2 + 0.25 * g**2))
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
        d_i = 0 * (pol[0] == 'x') + 1 * (pol[0] == 'y') + 2 * (pol[0] == 'z')
        d_o = 0 * (pol[1] == 'x') + 1 * (pol[1] == 'y') + 2 * (pol[1] == 'z')
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
    w_l, rep_l = count_deg(freqs_l)

    # Add the phonon bars to the figure with showing their degeneracy factors
    pltbar = plt.bar(w_l, -0.04, width=maxw / 100, color='k')
    for idx, rect in enumerate(pltbar):
        ax.text(rect.get_x() + rect.get_width() / 2., -0.1,
                str(int(rep_l[idx])), ha='center', va='bottom', rotation=0)

    # Remove the extra space and save the figure
    plt.tight_layout()
    plt.savefig(filename)

# Count the modes and their degeneracy factors


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
