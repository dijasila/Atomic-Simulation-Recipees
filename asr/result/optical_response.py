import typing
import numpy as np

from asr.core import ASRResult, prepare_result
from asr.result.core.exceptions import CentroSymmetric
from asr.extra_fluff import create_plot_simple
from asr.panels.createwebpanel import (
    OpticalWebpanel, InfraredWebpanel, ShgWebpanel, ShiftWebpanel,
)


@prepare_result
class OpticalResult(ASRResult):
    alphax_el: typing.List[complex]
    alphay_el: typing.List[complex]
    alphaz_el: typing.List[complex]
    plasmafreq_vv: typing.List[typing.List[float]]
    frequencies: typing.List[float]

    key_descriptions = {
        "alphax_el": "Optical polarizability (x) [Ang]",
        "alphay_el": "Optical polarizability (y) [Ang]",
        "alphaz_el": "Optical polarizability (z) [Ang]",
        "plasmafreq_vv": "Plasmafrequency tensor.",
        "frequencies": "Frequency grid [eV]."
    }

    formats = {"ase_webpanel": OpticalWebpanel}

    @staticmethod
    def polarizability(row, fx, fy, fz):
        import matplotlib.pyplot as plt

        def ylims(ws, data, wstart=0.0):
            i = abs(ws - wstart).argmin()
            x = data[i:]
            x1, x2 = x.real, x.imag
            y1 = min(x1.min(), x2.min()) * 1.02
            y2 = max(x1.max(), x2.max()) * 1.02
            return y1, y2

        def plot_polarizability(ax, frequencies, alpha_w, filename, direction):
            ax.set_title(f'Polarization: {direction}')
            ax.set_xlabel('Energy [eV]')
            ax.set_ylabel(r'Polarizability [$\mathrm{\AA}$]')
            ax.set_ylim(ylims(ws=frequencies, data=alpha_w, wstart=0.5))
            ax.legend()
            ax.set_xlim((0, 10))
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(filename)

        data = row.data.get('results-asr.polarizability.json')

        if data is None:
            return
        frequencies = data['frequencies']
        i2 = abs(frequencies - 50.0).argmin()
        frequencies = frequencies[:i2]
        alphax_w = data['alphax_w'][:i2]
        alphay_w = data['alphay_w'][:i2]
        alphaz_w = data['alphaz_w'][:i2]

        ax = plt.figure().add_subplot(111)
        ax1 = ax
        try:
            wpx = row.plasmafrequency_x
            if wpx > 0.01:
                alphaxfull_w = alphax_w - wpx**2 / (2 * np.pi *
                                                    (frequencies + 1e-9)**2)
                ax.plot(
                    frequencies,
                    np.real(alphaxfull_w),
                    '-',
                    c='C1',
                    label='real')
                ax.plot(
                    frequencies,
                    np.real(alphax_w),
                    '--',
                    c='C1',
                    label='real (interband)')
            else:
                ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
        except AttributeError:
            ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
        ax.plot(frequencies, np.imag(alphax_w), c='C0', label='imag')

        plot_polarizability(ax, frequencies, alphax_w, filename=fx,
                            direction='x')

        ax = plt.figure().add_subplot(111)
        ax2 = ax
        try:
            wpy = row.plasmafrequency_y
            if wpy > 0.01:
                alphayfull_w = alphay_w - wpy**2 / (2 * np.pi *
                                                    (frequencies + 1e-9)**2)
                ax.plot(
                    frequencies,
                    np.real(alphayfull_w),
                    '-',
                    c='C1',
                    label='real')
                ax.plot(
                    frequencies,
                    np.real(alphay_w),
                    '--',
                    c='C1',
                    label='real (interband)')
            else:
                ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')
        except AttributeError:
            ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')

        ax.plot(frequencies, np.imag(alphay_w), c='C0', label='imag')
        plot_polarizability(ax, frequencies, alphay_w, filename=fy,
                            direction='y')

        ax3 = plt.figure().add_subplot(111)
        ax3.plot(frequencies, np.real(alphaz_w), c='C1', label='real')
        ax3.plot(frequencies, np.imag(alphaz_w), c='C0', label='imag')
        plot_polarizability(ax3, frequencies, alphaz_w, filename=fz,
                            direction='z')

        return ax1, ax2, ax3


@prepare_result
class InfraredResult(ASRResult):
    alpha_wvv: typing.List[typing.List[typing.List[complex]]]
    omega_w: typing.List[float]
    alphax_lat: complex
    alphay_lat: complex
    alphaz_lat: complex
    alphax: complex
    alphay: complex
    alphaz: complex

    key_descriptions = {
        "alpha_wvv": "Lattice polarizability.",
        "omega_w": "Frequency grid [eV].",
        "alphax_lat": "Lattice polarizability at omega=0 (x-direction).",
        "alphay_lat": "Lattice polarizability at omega=0 (y-direction).",
        "alphaz_lat": "Lattice polarizability at omega=0 (z-direction).",
        "alphax": "Lattice+electronic polarizability at omega=0 (x-direction).",
        "alphay": "Lattice+electronic polarizability at omega=0 (y-direction).",
        "alphaz": "Lattice+electronic polarizability at omega=0 (z-direction).",
    }

    formats = {"ase_webpanel": InfraredWebpanel}

    @staticmethod
    def create_plot(row, *fnames):
        infrareddct = row.data['results-asr.infraredpolarizability.json']
        electrondct = row.data['results-asr.polarizability.json']
        phonondata = row.data['results-asr.phonons.json']
        maxphononfreq = phonondata['omega_kl'][0].max() * 1e3

        assert len(fnames) == 3
        for v, (axisname, fname) in enumerate(zip('xyz', fnames)):
            alpha_w = electrondct[f'alpha{axisname}_w']

            create_plot_simple(
                ndim=sum(row.toatoms().pbc),
                maxomega=maxphononfreq * 1.5,
                omega_w=infrareddct["omega_w"] * 1e3,
                alpha_w=alpha_w,
                alphavv_w=infrareddct["alpha_wvv"][:, v, v],
                omegatmp_w=electrondct["frequencies"] * 1e3,
                axisname=axisname,
                fname=fname)


@prepare_result
class ShgResult(ASRResult):
    freqs: typing.List[float]
    chi: typing.Dict[str, typing.Any]
    symm: typing.Dict[str, str]

    key_descriptions = {
        "freqs": "Pump photon energy [eV]",
        "chi": "Non-zero SHG tensor elements in SI units",
        "symm": "Symmetry relation of SHG tensor",
    }
    formats = {"ase_webpanel": ShgWebpanel}

    @staticmethod
    def plot_shg(row, *filename):
        import matplotlib.pyplot as plt
        import os
        from pathlib import Path
        from textwrap import wrap

        # Read the data from the disk
        data = row.data.get('results-asr.shg.json')
        gap = row.get('gap_dir')
        atoms = row.toatoms()
        pbc = atoms.pbc.tolist()
        nd = np.sum(pbc)

        # Remove the files if it already exists
        for fname in filename:
            if Path(fname).is_file():
                os.remove(fname)

        # Plot the data and add the axis labels
        sym_chi = data['symm']
        if len(sym_chi) == 1:
            raise CentroSymmetric
        chi = data['chi']

        if not chi:
            return
        w_l = data['freqs']
        fileind = 0
        axes = []

        for pol in sorted(chi.keys()):
            # Make the axis and add y=0 axis
            shg = chi[pol]
            ax = plt.figure().add_subplot(111)
            ax.axhline(y=0, color='k')

            # Add the bandgap
            bg = gap
            if bg is not None:
                ax.axvline(x=bg, color='k', ls='--')
                ax.axvline(x=bg / 2, color='k', ls='--')
                maxw = min(np.ceil(2.0 * bg), 7)
            else:
                maxw = 7

            # Plot the data
            amp_l = shg
            amp_l = amp_l[w_l < maxw]
            ax.plot(w_l[w_l < maxw], np.real(amp_l), '-', c='C0', label='Re')
            ax.plot(w_l[w_l < maxw], np.imag(amp_l), '-', c='C1', label='Im')
            ax.plot(w_l[w_l < maxw], np.abs(amp_l), '-', c='C2', label='Abs')

            # Set the axis limit
            ax.set_xlim(0, maxw)
            relation = sym_chi.get(pol)
            if not (relation is None):
                figtitle = '$' + '$\n$'.join(wrap(relation, 40)) + '$'
                ax.set_title(figtitle)
            ax.set_xlabel(r'Pump photon energy $\hbar\omega$ [eV]')
            polstr = f'{pol}'
            if nd == 2:
                ax.set_ylabel(r'$\chi^{(2)}_{' + polstr + r'}$ [nm$^2$/V]')
            else:
                ax.set_ylabel(r'$\chi^{(2)}_{' + polstr + r'}$ [nm/V]')
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

            # Add the legend
            ax.legend(loc='best')

            # Remove the extra space and save the figure
            plt.tight_layout()
            plt.savefig(filename[fileind])
            fileind += 1
            axes.append(ax)
            plt.close()

        # Now make the polarization-resolved plot
        psi = np.linspace(0, 2 * np.pi, 201)
        selw = 0
        wind = np.argmin(np.abs(w_l - selw))
        if Path('shgpol.npy').is_file():
            os.remove('shgpol.npy')
        chipol = ShgResult.calc_polarized_shg(
            sym_chi, chi,
            wind=[wind], theta=0, phi=0,
            pte=np.sin(psi), ptm=np.cos(psi), outname=None, outbasis='pol')
        ax = plt.subplot(111, projection='polar')
        ax.plot(psi, np.abs(chipol[0]), 'C0', lw=1.0)
        ax.plot(psi, np.abs(chipol[1]), 'C1', lw=1.0)
        # Set the y limits
        ax.grid(True)
        rmax = np.amax(np.abs(chipol))
        if np.abs(rmax) < 1e-6:
            rmax = 1e-4
            ax.plot(0, 0, 'o', color='b', markersize=5)
        ax.set_rlim(0, 1.2 * rmax)
        ax.set_rgrids([rmax], fmt=r'%4.2g')
        labs = [r'  $\theta=0$', '45', '90', '135', '180', '225', '270', '315']
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=labs)

        # Put a legend below current axis
        ax.legend([
            r'Parallel: |$\chi^{(2)}_{\theta \theta \theta}$|',
            r'Perpendicular: |$\chi^{(2)}_{(\theta+90)\theta \theta}$|'],
            loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True,
            ncol=2)

        # Remove the extra space and save the figure
        plt.tight_layout()
        plt.savefig(filename[fileind])
        axes.append(ax)

        return tuple(axes)

    @staticmethod
    def make_full_chi(sym_chi, chi_dict):

        if len(sym_chi) == 1:
            return 0

        # Make the full chi from its symmetries
        for pol in sorted(sym_chi.keys()):
            if pol != 'zero':
                chidata = chi_dict[pol]
                nw = len(chidata)
        chi_vvvl = np.zeros((3, 3, 3, nw), complex)
        for pol in sorted(sym_chi.keys()):
            relation = sym_chi.get(pol)
            if pol == 'zero':
                if relation != '':
                    for zpol in relation.split('='):
                        ind = ['xyz'.index(zpol[ii]) for ii in range(3)]
                        chi_vvvl[ind[0], ind[1], ind[2]] = np.zeros(nw,
                                                                    complex)
            else:
                chidata = chi_dict[pol]
                chidata = chidata[1]
                for zpol in relation.split('='):
                    if zpol[0] == '-':
                        ind = ['xyz'.index(zpol[ii + 1]) for ii in range(3)]
                        chi_vvvl[ind[0], ind[1], ind[2]] = -chidata
                    else:
                        ind = ['xyz'.index(zpol[ii]) for ii in range(3)]
                        chi_vvvl[ind[0], ind[1], ind[2]] = chidata

        return chi_vvvl

    @staticmethod
    def calc_polarized_shg(sym_chi, chi_dict, wind=[1], theta=0.0, phi=0.0,
                           pte=[1.0], ptm=[0.0], E0=[1.0], outname=None,
                           outbasis='pol'):
        # Check the input arguments
        pte = np.array(pte)
        ptm = np.array(ptm)
        E0 = np.array(E0)
        assert np.all(
            np.abs(pte)**2 + np.abs(ptm)**2) == 1, \
            '|pte|**2+|ptm|**2 should be one.'
        assert len(pte) == len(ptm), 'Size of pte and ptm should be the same.'

        # Useful variables
        costh = np.cos(theta)
        sinth = np.sin(theta)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        nw = len(wind)
        npsi = len(pte)

        # Transfer matrix between (x y z)/(atm ate k) unit vectors basis
        if theta == 0:
            transmat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            transmat = [[cosphi * costh, sinphi * costh, -sinth],
                        [-sinphi, cosphi, 0],
                        [sinth * cosphi, sinth * sinphi, costh]]
        transmat = np.array(transmat)

        # Get the full chi tensor
        chi_vvvl = ShgResult.make_full_chi(sym_chi, chi_dict)

        # Check the E0
        if len(E0) == 1:
            E0 = E0 * np.ones(nw)

        # in xyz coordinate
        Einc = np.zeros((3, npsi), dtype=complex)
        for v1 in range(3):
            Einc[v1] = (pte * transmat[0][v1] + ptm * transmat[1][v1])

        # Loop over components
        chipol = np.zeros((3, npsi, nw), dtype=complex)
        for ii, wi in enumerate(wind):
            for ind in range(27):
                v1, v2, v3 = int(ind / 9), int((ind % 9) / 3), (ind % 9) % 3
                if chi_vvvl[v1, v2, v3, wi] != 0.0:
                    chipol[v1, :, ii] += chi_vvvl[v1, v2, v3, wi] * \
                                         Einc[v2, :] * Einc[v3, :] * E0[ii]**2

        # Change the output basis if needed, and return
        if outbasis == 'xyz':
            chipol_new = chipol
        elif outbasis == 'pol':
            chipol_new = np.zeros((3, npsi, nw), dtype=complex)
            for ind, wi in enumerate(wind):
                chipol[:, :, ind] = np.dot(transmat.T, chipol[:, :, ind])
                chipol_new[0, :, ind] = (chipol[0, :, ind] *
                                         pte + chipol[1, :, ind] * ptm)
                chipol_new[1, :, ind] = (-chipol[0, :, ind] *
                                         ptm + chipol[1, :, ind] * pte)

        else:
            raise NotImplementedError

        return chipol_new


@prepare_result
class ShiftResult(ASRResult):
    freqs: typing.List[float]
    sigma: typing.Dict[str, typing.Any]
    symm: typing.Dict[str, str]

    key_descriptions = {
        "freqs": "Photon energy [eV]",
        "sigma": "Non-zero shift conductivity tensor elements in SI units",
        "symm": "Symmetry relation of shift conductivity tensor",
    }
    formats = {"ase_webpanel": ShiftWebpanel}

    @staticmethod
    def plot_shift(row, *filename):
        import matplotlib.pyplot as plt
        import os
        from pathlib import Path
        from textwrap import wrap

        # Read the data from the disk
        data = row.data.get('results-asr.shift.json')
        gap = row.get('gap_dir_nosoc')
        atoms = row.toatoms()
        pbc = atoms.pbc.tolist()
        nd = np.sum(pbc)
        if data is None:
            return

        # Remove the files if it already exists
        for fname in filename:
            if Path(fname).is_file():
                os.remove(fname)

        # Plot the data and add the axis labels
        sym_chi = data['symm']
        if len(sym_chi) == 1:
            raise CentroSymmetric
        sigma = data['sigma']

        if not sigma:
            return
        w_l = data['freqs']
        fileind = 0
        axes = []

        for pol in sorted(sigma.keys()):
            # Make the axis and add y=0 axis
            shift_l = sigma[pol]
            ax = plt.figure().add_subplot(111)
            ax.axhline(y=0, color='k')

            # Add the bandgap
            if gap is not None:
                ax.axvline(x=gap, color='k', ls='--')

            # Plot the data
            ax.plot(w_l, np.real(shift_l), '-', c='C0', )

            # Set the axis limit
            ax.set_xlim(0, np.max(w_l))
            relation = sym_chi.get(pol)
            if not (relation is None):
                figtitle = '$' + '$\n$'.join(wrap(relation, 40)) + '$'
                ax.set_title(figtitle)
            ax.set_xlabel(r'Energy [eV]')
            polstr = f'{pol}'
            if nd == 2:
                ax.set_ylabel(
                    r'$\sigma^{(2)}_{' + polstr + r'}$ [nm$\mu$A/V$^2$]')
            else:
                ax.set_ylabel(
                    r'$\sigma^{(2)}_{' + polstr + r'} [$\mu$A/V$^2$]')
            ax.ticklabel_format(axis='both', style='plain', scilimits=(-2, 2))

            # Remove the extra space and save the figure
            plt.tight_layout()
            plt.savefig(filename[fileind])
            fileind += 1
            axes.append(ax)
            plt.close()

        return tuple(axes)
