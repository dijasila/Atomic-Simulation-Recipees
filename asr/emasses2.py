"""Effective masses - version 117-b."""
from __future__ import annotations

import json
from math import pi
from pathlib import Path
from typing import TypedDict, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from ase.units import Bohr, Ha

from asr.core import ASRResult, command, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  fig, make_panel_description)
from asr.magnetic_anisotropy import get_spin_axis
from asr.utils.ndpoly import PolyFit, nterms

if TYPE_CHECKING:
    from gpaw.new.ase_interface import GPAW


panel_description = make_panel_description(
    """The effective mass tensor represents the second derivative of the band
energy w.r.t. wave vector at a band extremum. The effective masses of the
valence bands (VB) and conduction bands (CB) are obtained as the eigenvalues
of the mass tensor.  Spin–orbit interactions are included.""")


def webpanel(result, row, key_descriptions):
    rows = []
    for data in [result.cbm_mass, result.vbm_mass]:
        for n, mass in enumerate(data['mass_w'], start=1):
            kind = data['kind']
            rows.append([f'{kind}, direction {n}',
                         f'{mass:.2f} m<sub>0</sub>'])
    table = {'type': 'table',
             'header': ['', 'value'],
             'rows': rows}

    parameter_description = entry_parameter_description(
        row.data,
        'asr.emasses2')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Effective masses',
                                     description=title_description),
             'columns': [[fig('cbm-mass.png'), table],
                         [fig('vbm-mass.png')]],
             'plot_descriptions':
                 [{'function': mass_plots,
                   'filenames': ['cbm-mass.png',
                                 'vbm-mass.png']}]}

    return [panel]


def mass_plots(row, *filenames):
    result = row.data.get('results-asr.emasses2.json')
    cbm = result.cbm_mass
    vbm = result.vbm_mass

    x_p = np.linspace(-0.2, 0.2, 51)
    y_bwp = []
    height = 0.0
    offset = -vbm['energy']
    for data in [cbm, vbm]:
        y_wp = []
        dir_wv = np.array(data['direction_wv'])
        fit = PolyFit.from_coefs(coefs=data['fit_data_i'],
                                 order=4,
                                 ndims=len(dir_wv))
        for dir_v in dir_wv:
            y_p = [fit.value(x * dir_v) + offset for x in x_p]
            height = max(height, np.ptp(y_p))
            y_wp.append(y_p)
        y_bwp.append(y_wp)

    height *= 1.1
    plots = []
    for data, y_wp, filename in zip([cbm, vbm], y_bwp, filenames):
        dir_wv = data['direction_wv']
        fig, ax = plt.subplots()
        for n, (dir_v, y_p, ls) in enumerate(zip(dir_wv,
                                                 y_wp,
                                                 ['-', '--', ':']),
                                             1):
            d = ','.join(f'{d:.2f}' for d in dir_v)
            ax.plot(x_p, y_p,
                    ls=ls,
                    label=f'direction {n}: [{d}]')
        plots.append(ax)
        ax.legend()
        y0 = (np.max(y_wp) + np.min(y_wp)) / 2
        ax.axis(ymin=y0 - height / 2, ymax=y0 + height / 2)
        ax.set_xlabel(r'$\Delta k [1/Å]$')
        ax.set_ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]')
        fig.tight_layout()
        fig.savefig(filename)
    return plots


class GPAWEigenvalueCalculator:
    def __init__(self,
                 calc: GPAW,
                 theta: float = None,  # degrees
                 phi: float = None):  # degrees
        """Do SOC calculation.

        Parameters
        ----------
        calc:
            GPAW ground-state object
        theta:
            Polar angle in degrees.
        phi:
            Azimuthal angle in degrees.

        Returns
        -------
        * k-point vectors
        * eigenvalues
        * PAW-projections
        * spin-projections
        """
        self.calc = calc
        self.theta = theta
        self.phi = phi

        if theta is None and phi is None:
            eig_Kn, fermilevel = self._eigs()
        else:
            assert theta is not None and phi is not None
            eig_Kn, fermilevel = self._soc_eigs()

        self.nocc = (eig_Kn[0] < fermilevel).sum()

        kshape = tuple(calc.kd.N_c)
        eig_ijkn = eig_Kn.reshape(kshape + (-1,))
        self.vbm_ijk = eig_ijkn[..., self.nocc - 1].copy()
        self.cbm_ijk = eig_ijkn[..., self.nocc].copy()

        self.kpt_ijkc = calc.kd.bzk_kc.reshape(kshape + (3,))

    def band(self, kind, kpt_xv):
        from gpaw.spinorbit import soc_eigenstates

        kpt_xc = kpt_xv @ self.cell_cv.T / (2 * pi)
        nsc_calc = self.calc.fixed_density(
            kpts=kpt_xc,
            symmetry='off',
            convergence={'bands': 0 if kind == 'vbm' else -1},
            txt=f'{kind}.txt')
        if self.theta is None:
            nkpts = len(kpt_xv)
            nspins = self.calc.get_number_of_spins()
            eig_skn = np.array(
                [[self.calc.get_eigenvalues(kpt=kpt, spin=spin)
                  for kpt in range(nkpts)]
                 for spin in range(nspins)])
            eig_kns = eig_skn.transpose((1, 2, 0))
            eig_kn = eig_kns.reshape((nkpts, -1)).sort(axis=1)
        else:
            states = soc_eigenstates(nsc_calc, theta=theta, phi=phi)
            eig_kn = states.eigenvalues()
        return eig_kn

    def _eigs(self):
        kd = self.calc.wfs.kd
        nibzkpts = self.calc.get_number_of_ibz_k_points()
        nspins = self.calc.get_number_of_spins()
        eig_skn = np.array(
            [[self.calc.get_eigenvalues(kpt=kpt, spin=spin)
              for kpt in range(nibzkpts)]
             for spin in range(nspins)])
        eig_kns = eig_skn.transpose((1, 2, 0))
        eig_kn = eig_kns.reshape((nibzkpts, -1)).sort(axis=1)
        eig_Kn = eig_kn[kd.bz2ibz_k]
        fermilevel = self.calc.get_fermi_level()
        return eig_Kn, fermilevel

    def _soc_eigs(self):
        # Extract SOC eigenvalues:
        from gpaw.spinorbit import soc_eigenstates

        states = soc_eigenstates(self.calc, theta=self.theta, phi=self.phi)
        kd = self.calc.wfs.kd
        kpt_Kc = np.array([kd.bzk_kc[wf.bz_index]
                           for wf in states])
        assert (kpt_Kc == kd.bzk_kc).all()
        eig_Kn = states.eigenvalues()
        fermilevel = states.fermi_level
        return eig_Kn, fermilevel


def extract_soc_stuff_from_gpaw_calculation(calc,
                                            npoints: int = 7,
                                            kspan: float = 0.1,  # Ang^-1
                                            nblock: int = 4,
                                            theta: float = 0.0,  # degrees
                                            phi: float = 0.0,  # degrees
                                            ) -> dict[str,
                                                      dict[str, np.ndarray]]:
    """Do SOC calculation.

    Parameters
    ----------
    calc:
        GPAW ground-state object
    theta:
        Polar angle in degrees.
    phi:
        Azimuthal angle in degrees.

    Returns
    -------
    * k-point vectors
    * eigenvalues
    * PAW-projections
    * spin-projections
    """
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.mpi import world

    datafile = Path('emass-data.json')
    if datafile.is_file():
        data = json.loads(datafile.read_text())
        # Convert to (complex) ndarrays:
        return {kind: {name: np.array(array).view(complex)
                       if name == 'proj_ijknI' else
                       np.array(array)
                       for name, array in things.items()}
                for kind, things in data.items()}

    # Extract SOC eigenvalues:
    states = soc_eigenstates(calc, theta=theta, phi=phi)
    kd = calc.wfs.kd
    k_Kc = np.array([kd.bzk_kc[wf.bz_index]
                     for wf in states])
    eig_Kn = np.array([wf.eig_m
                       for wf in states])
    fermilevel = states.fermi_level
    nocc = (eig_Kn[0] < fermilevel).sum()
    cell_cv = calc.atoms.cell

    k_Kv = k_Kc @ np.linalg.inv(cell_cv).T * 2 * pi
    kspan2 = ((k_Kv[1:] - k_Kv[0])**2).sum(1).min()**0.5

    # Create box of k-points centered on (0, 0, 0):
    K_c = kd.N_c
    dk_vk = [np.linspace(-kspan / 2, kspan / 2, npoints) if K > 1 else
             np.array([0.0])
             for K in K_c]
    shape = tuple(len(dk_k) for dk_k in dk_vk)
    dk_kv = np.empty(shape + (3,))
    dk_kv[..., 0] = dk_vk[0][:, np.newaxis, np.newaxis]
    dk_kv[..., 1] = dk_vk[1][:, np.newaxis]
    dk_kv[..., 2] = dk_vk[2]
    dk_kv.shape = (-1, 3)

    # Find band extrema and zoom in:
    data = {}
    for kind, eig_K in [('vbm', -eig_Kn[:, nocc - 1]),
                        ('cbm', eig_Kn[:, nocc])]:
        K = eig_K.argmin()
        k_v = k_Kv[K]
        k_kv = k_v + dk_kv
        k_kc = k_kv @ cell_cv.T / (2 * pi)

    if world.rank == 0:
        # We need to convert complex to two floats before converting to
        # json:
        datafile.write_text(
            json.dumps(
                {kind: {name: array.copy().view(float).tolist()
                        for name, array in things.items()}
                 for kind, things in data.items()},
                indent=1))

    return data


class MassDict(TypedDict):
    kind: str
    k_v: list[float]
    mass_w: list[float]
    direction_wv: list[list[float]]
    energy: float
    max_fit_error: float
    fit_data_i: list[float]


@prepare_result
class EMassesResult(ASRResult):
    vbm_mass: MassDict
    cbm_mass: MassDict

    key_descriptions = {
        'vbm_mass': 'Mass data for VBM',
        'cbm_mass': 'Mass data for CBM'}

    formats = {'ase_webpanel': webpanel}


@command('asr.emasses2',
         requires=['gs.gpw'],
         dependencies=['asr.gs'])
def main() -> ASRResult:
    """Find effective masses."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw')

    theta, phi = get_spin_axis()

    data = extract_soc_stuff_from_gpaw_calculation(calc, theta=theta, phi=phi)

    extrema = []
    for kind in ['vbm', 'cbm']:
        massdata = find_mass(**data[kind], kind=kind)
        extrema.append(massdata)

    vbm, cbm = extrema
    return EMassesResult.fromdata(vbm_mass=vbm, cbm_mass=cbm)


class BadFitError(ValueError):
    """Bad fit to data."""


def basin(eig_ijk, i, j, k):
    """...

    >>> eigs = np.array([[[1, 0, 1, 0.5]]])
    >>> basin(eigs, 0, 0, 1)
    """
    include = {(i, j, k): True}
    mask_ijk = np.empty(eig_ijk.shape, bool)
    for i0 in range(eig_ijk.shape[0]):
        for j0 in range(eig_ijk.shape[1]):
            for k0 in range(eig_ijk.shape[2]):
                ok = _basin(eig_ijk, i0, j0, k0, include)
                mask_ijk[i0, j0, k0] = ok
    return mask_ijk


def _basin(eig_ijk, i, j, k, include):
    ijk = (i, j, k)
    if ijk in include:
        return include[ijk]

    candidates = [(eig_ijk[i, j, k], ijk)]
    x_c = list(ijk)
    for d in [-1, 1]:
        for c in [0, 1, 2]:
            x_c[c] += d
            if 0 <= x_c[c] < eig_ijk.shape[c]:
                candidates.append((eig_ijk[x_c[0], x_c[1], x_c[2]],
                                   tuple(x_c)))
            x_c[c] -= d
    _, ijk0 = min(candidates)
    if ijk0 == ijk:
        ok = False
    else:
        ok = basin(eig_ijk, *ijk0, include)
    include[ijk] = ok
    return ok


def find_mass(kpt_ijkv: np.ndarray,
              eig_ijk: np.ndarray,
              kind: str,
              eigcalc=None,
              maxlevels: int = 1) -> MassDict:
    if kind == 'vbm':
        eig_ijk = -eig_ijk

    shape = eig_ijk.shape
    i, j, k = np.unravel_index(eig_ijk.ravel().argmin(), shape)
    kpt_v = kpt_ijkv[i, j, k]
    mask = basin(eig_ijk, i, j, k).ravel()
    eig_x = eig_ijk.ravel()[mask]
    kpt_xv = kpt_ijkv.reshape((-1, 3))[mask]

    axes = [c for c, size in enumerate(eig_ijk.shape) if size > 1]
    kpt_xv = kpt_xv[:, axes]

    if len(eig_x) > 1.25 * nterms(4, len(axes)):
        try:
            kpt_v, energy, mass_w, direction_wv, error_x, fit = fit_band(
                kpt_xv, eig_x)
        except FitError:
            pass
        else:
            fit_error = abs(error_x).max()
            if fit_error < 0.02 * eig_x.ptp():
                if kind == 'vbm':
                    energy *= -1
                    fit.coefs *= -1
                return MassDict(kind=kind,
                                k_v=k_v.tolist(),
                                energy=energy,
                                mass_w=mass_w.tolist(),
                                direction_wv=direction_wv.tolist(),
                                max_fit_error=fit_error,
                                fit_data_i=fit.coefs.tolist())
    if maxlevels == 1:
        raise ValueError

    kpt_xv = kpt_ijkv.reshape(shape + (3,))
    knndist = ((kpt_xv[1:] - kpt_xv[0])**2).sum(1).min()**0.5
    kspan = 2.5 * knndist

    # Create box of k-points centered on k_v:
    kpt_vx = [np.linspace(kpt - kspan / 2, kpt + kspan / 2, npoints)
              if npoints > 1
              else np.array([0.0])
              for kpt, npoints in zip(kpt_v, shape)]
    kpt_ijkv = np.empty(shape + (3,))
    kpt_ijkv[..., 0] = kpt_vx[0][:, np.newaxis, np.newaxis]
    kpt_ijkv[..., 1] = kpt_vx[1][:, np.newaxis]
    kpt_ijkv[..., 2] = kpt_vx[2]

    eig_ijk = eigcalc.band(kpt_ijkv, kind)
    return find_mass(kpt_ijkv, eig_ijk)


class FitError(ValueError):
    ...


def fit_band(k_xv: np.ndarray,
             eig_x: np.ndarray) -> tuple[np.ndarray,
                                         float,
                                         np.ndarray,
                                         np.ndarray,
                                         np.ndarray,
                                         PolyFit]:
    kmin_v = k_xv[eig_x.argmin()].copy()
    dk_xv = k_xv - kmin_v
    fit = PolyFit(dk_xv, eig_x, order=4)
    error_x = np.array([fit.value(k_v) - e for k_v, e in zip(dk_xv, eig_x)])
    hessian_vv = fit.hessian(np.zeros_like(kmin_v))
    eval_w = np.linalg.eigvalsh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise FitError('Not a minimum')

    dkmin_v = fit.find_minimum()
    emin = fit.value(dkmin_v)
    hessian_vv = fit.hessian(dkmin_v)
    eval_w, evec_vw = np.linalg.eigh(hessian_vv)
    mass_w = Bohr**2 * Ha / eval_w
    assert (mass_w > 0.0).all()

    # Centered fit around minumum:
    fit = PolyFit(dk_xv - dkmin_v, eig_x, order=4)

    return dkmin_v + kmin_v, emin, mass_w, evec_vw.T, error_x, fit


if __name__ == '__main__':
    main.cli()
