from asr.core import command


def webpanel(row, key_descriptions):
    from asr.browser import fig

    panel = {
        "title": "Raman spectrum (RPA)",
        "columns": [[fig("Raman.png")], []],
        "plot_descriptions": [{"function": raman, "filenames": ["Raman.png"]}],
        "sort": 20,
    }

    return [panel]


@command("asr.raman", webpanel=webpanel)
def main():
    raise NotImplementedError


def raman(row, filename):
    # Import the required modules
    import numpy as np
    import matplotlib.pyplot as plt

    # All required settings
    params = {
        "broadening": 10.0,  # in cm^-1
        "wavelength": 532.0,  # in nm
        "polarization": ["xx", "yy", "zz"],
    }
    # 'polarization': ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']}

    # Read the data from the disk
    data = row.data.get("results-asr.raman.json")

    # If no data, return
    if data is None:
        return

    # Lorentzian function definition
    def lor(w, g):
        lor = 0.5 * g / (np.pi * ((w.real) ** 2 + 0.25 * g ** 2))
        return lor

    # Compute spectrum based on a set of resonances
    def calcspectrum(wlist, rlist, ww, gamma=10, shift=0):
        rr = np.zeros(np.size(ww))
        for wi, ri in zip(wlist, rlist):
            # Only add the modes with positive frequencies (stable)
            if wi >= 0:
                rr = rr + np.abs(ri) ** 2 * lor(ww - wi - shift, gamma)

        # Normalize it to 1 and return it
        # rr = rr/np.max(rr)
        return rr

    # Make a latex type formula
    def getformula(matstr):
        matformula = r""
        for ch in matstr:
            if ch.isdigit():
                matformula += "$_" + ch + "$"
            else:
                matformula += ch
        return matformula

    # Set the variables and parameters
    wavelength_w = data["wavelength_w"]
    freqs_l = data["freqs_l"]
    amplitudes_vvwl = data["amplitudes_vvwl"]
    selpol = params["polarization"]
    gamma = params["broadening"]

    # If the wavelength was not found, return
    waveind = int(np.where(wavelength_w == params["wavelength"])[0])
    if not waveind:
        return

    # Check the data to be consistent
    ampshape = amplitudes_vvwl.shape
    freqshape = len(freqs_l)
    waveshape = len(wavelength_w)
    if (
        (ampshape[0] != 3)
        or (ampshape[1] != 3)
        or (ampshape[2] != waveshape)
        or (ampshape[3] != freqshape)
    ):
        return

    # Make the spectrum
    maxw = min([int(np.max(freqs_l) + 200), int(1.2 * np.max(freqs_l))])
    minw = -maxw / 100
    ww = np.linspace(minw, maxw, 2 * maxw)
    rr = {}
    maxr = np.zeros(len(selpol))
    for ii, pol in enumerate(selpol):
        d_i = 0 * (pol[0] == "x") + 1 * (pol[0] == "y") + 2 * (pol[0] == "z")
        d_o = 0 * (pol[1] == "x") + 1 * (pol[1] == "y") + 2 * (pol[1] == "z")
        rr[pol] = calcspectrum(
            freqs_l,
            amplitudes_vvwl[d_i, d_o, waveind],
            ww,
            gamma=gamma,
            shift=0,
        )
        maxr[ii] = np.max(rr[pol])

    # Set the defaults for plot
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams[
        "font.serif"
    ]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["font.size"] = 11
    plt.rcParams["lines.markersize"] = 5
    plt.rcParams["lines.markeredgecolor"] = "k"
    plt.rcParams["lines.markeredgewidth"] = 0.5
    plt.style.use("seaborn-bright")

    # Make the figure panel and add y=0 axis
    ax = plt.figure().add_subplot(111)
    ax.axhline(y=0, color="k")

    # Plot the data and add the axis labels
    for ipol, pol in enumerate(selpol):
        ax.plot(ww, rr[pol] / np.max(maxr), c="C" + str(ipol), label=pol)
    figtitle = "Raman spectrum of {} at {} nm wavelength".format(
        getformula(row.formula), params["wavelength"]
    )
    ax.set_title(figtitle)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Raman intensity (a.u.)")
    ax.set_ylim((-0.2, 1.1))
    ax.set_yticks([])
    ax.set_xlim((minw, maxw))

    # Add the legend to figure
    ax.legend()

    # Count the modes and their degeneracy factors
    freq_err = 2  # assume modes that are closer as one mode, in cm^-1
    w_l = [freqs_l[0]]  # the degeneracy factor for modes
    rep_l = [1]
    for wss in freqs_l[1:]:
        ind = len(w_l) - 1
        if np.abs(w_l[ind] - wss) > freq_err:
            w_l.append(wss)
            rep_l.append(1)
        else:
            rep_l[ind] += 1
    w_l = np.array(w_l)
    rep_l = np.array(rep_l)

    # Add the phonon bars to the figure with showing their degeneracy factors
    pltbar = plt.bar(w_l, -0.1, width=maxw / 100, color="b")
    for idx, rect in enumerate(pltbar):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            -0.17,
            str(int(rep_l[idx])),
            ha="center",
            va="bottom",
            rotation=0,
        )

    # Remove the extra space and save the figure
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == '__main__':
    main.cli()
