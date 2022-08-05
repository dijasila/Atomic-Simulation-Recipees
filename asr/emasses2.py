"""Effective masses - version 117-b."""
from __future__ import annotations

import json
from math import pi
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from ase.units import Bohr, Ha

from asr.core import ASRResult, command, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  fig, make_panel_description)
from asr.magnetic_anisotropy import get_spin_axis
from asr.utils.ndpoly import PolyFit

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


def extract_soc_stuff_from_gpaw_calculation(calc,
                                            npoints: int = 6,
                                            kspan: float = 0.1,  # Ang^-1
                                            nblock: int = 4,
                                            theta: float = 0.0,  # degrees
                                            phi: float = 0.0,  # degrees
                                            ) -> tuple[np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray]:
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
        return data

    # Extract SOC eigenvalues:
    states = soc_eigenstates(calc, theta=theta, phi=phi)
    kd = calc.wfs.kd
    k_Kc = np.array([kd.bzk_kc[wf.bz_index]
                     for wf in states])
    eig_Kn = np.array([wf.eig_m
                       for wf in states])
    fermilevel = states.fermi_level
    nocc = (eig_Kn[0] < fermilevel).sum()

    K_c = kd.N_c
    cell_cv = calc.atoms.cell

    # Create box of k-points centered on (0, 0, 0):
    dk_vk = [np.linspace(-kspan / 2, kspan / 2, npoints) if K > 1 else
             np.array([0.0])
             for K in K_c]
    dk_kv = (dk_vk[0][:, np.newaxis, np.newaxis] *
             dk_vk[1][:, np.newaxis] *
             dk_vk[2])

    # Break symmetries:
    dk = kspan / npoints * 0.05
    dk_kv += np.random.default_rng().uniform(-dk, dk, dk_kv.shape)
    world.broadcast(dk_kv, 0)

    # Find band extrema and zoom in:
    data = {}
    for kind, eig_K in [('vbm', -eig_Kn[nocc - 1]),
                        ('cbm', eig_Kn[nocc])]:
        K = eig_K.argmin()
        k_c = k_Kc[K]
        k_v = k_c @ np.linalg.inv(cell_cv).T * 2 * pi
        k_kv = k_v + dk_kv
        k_kc = k_kv @ cell_cv.T / (2 * pi)

        nsc_calc = calc.fixed_density(
            kpts=k_kc,
            symmetry='off',
            txt=f'{kind}.txt')
        states = soc_eigenstates(nsc_calc, theta=theta, phi=phi)
        eig_kn = np.array([wf.eig_m
                           for wf in states])
        proj_knI = np.array([wf.projections.matrix.array
                             for wf in states])

        if kind == 'vbm':
            bands = slice(max(nocc - nblock, 0), nocc)
        else:
            bands = slice(nocc, nocc + nblock)

        eig_kn = eig_kn[:, bands]
        proj_knI = proj_knI[:, bands]

        def fix(a_kx: np.ndarray) -> list:
            """Fix shape and prepare for json."""
            return a_kx.reshape(tuple(K_c) + a_kx.shape[1:]).tolist()

        data[kind] = {'k_ijkv': fix(k_kv),
                      'eig_ijkn': fix(eig_kn),
                      'proj_ijknI': fix(proj_knI)}

    if world.rank == 0:
        datafile.write_text(json.dumps(data))

    return data


def connect(eig_ijkn: np.ndarray,
            fingerprint_ijknx: np.ndarray) -> np.ndarray:
    """Reorder eigenvalues to give connected bands."""
    eig_ijkn = eig_ijkn.copy()
    K1, K2, K3, N = eig_ijkn.shape
    for k1 in range(K1 - 1):
        con2(eig_ijkn[k1:k1 + 2, 0, 0],
             fingerprint_ijknx[k1:k1 + 2, 0, 0])

    for k1 in range(K1):
        for k2 in range(K2 - 1):
            con2(eig_ijkn[k1, k2:k2 + 2, 0],
                 fingerprint_ijknx[k1, k2:k2 + 2, 0])

    for k1 in range(K1):
        for k2 in range(K2):
            for k3 in range(K3 - 1):
                con2(eig_ijkn[k1, k2, k3:k3 + 2],
                     fingerprint_ijknx[k1, k2, k3:k3 + 2])
    return eig_ijkn


def con2(e_kn: np.ndarray,
         fingerprint_knx: np.ndarray) -> list[int]:
    """Connect 2 k-points."""
    K, N = e_kn.shape
    assert K == 2

    ovl_n1n2 = abs(fingerprint_knx[0] @ fingerprint_knx[1].conj().T)

    n2_n1 = []
    n1_n2: dict[int, int] = {}
    for n1 in range(N):
        n2 = ovl_n1n2[n1].argmax()
        ovl_n1n2[:, n2] = -1.0
        n2_n1.append(n2)
        n1_n2[n2] = n1

    fingerprint2_nx = fingerprint_knx[1].copy()

    e2_n = e_kn[1, n2_n1]
    fingerprint2_nx = fingerprint2_nx[n2_n1]
    e_kn[1] = e2_n
    fingerprint_knx[1] = fingerprint2_nx

    return n2_n1


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
         requires=['pdos.gpw'],
         dependencies=['asr.pdos'])
def main() -> ASRResult:
    """Find effective masses."""
    from gpaw import GPAW
    calc = GPAW('pdos.gpw')

    theta, phi = get_spin_axis()

    data = extract_soc_stuff_from_gpaw_calculation(calc, theta=theta, phi=phi)

    extrema = []
    for kind in ['vbm', 'cbm']:
        k_ijkv = np.array(data[kind]['k_ijkv'])
        e_ijkn = np.array(data[kind]['e_ijkn'])
        proj_ijknI = np.array(data[kind]['proj_ijknI'])
        massdata = find_mass(k_ijkv, e_ijkn, proj_ijknI, kind)
        extrema.append(massdata)

    vbm, cbm = extrema
    return EMassesResult.fromdata(vbm_mass=vbm, cbm_mass=cbm)


class BadFitError(ValueError):
    """Bad fit to data."""


def find_mass(k_ijkv: np.ndarray,
              eig_ijkn: np.ndarray,
              proj_ijknI: np.ndarray,
              kind: str) -> MassDict:
    if kind == 'vbm':
        eig_ijkn = -eig_ijkn[:, :, :, ::-1]

    # Connect bands:
    eig_ijkn = connect(eig_ijkn, proj_ijknI)

    # Find lowest band:
    _, _, _, n = np.unravel_index(eig_ijkn.argmin(), eig_ijkn.shape)
    eig_k = eig_ijkn[:, :, :, n].ravel()

    axes = [c for c, size in enumerate(k_ijkv.shape[:3]) if size > 1]
    k_kv = k_ijkv.reshape((-1, 3))[:, axes]

    k_v, energy, mass_w, direction_wv, error_k, fit = fit_band(k_kv, eig_k)

    fit_error = abs(error_k).max()
    if fit_error > max(0.02 * eig_k.ptp(), 0.001):
        raise BadFitError(f'Error: {fit_error} eV')

    error = abs(energy - eig_ijkn.min())
    if error > 0.1:
        raise BadFitError(f'Error: {error} eV')

    if kind == 'vbm':
        energy *= -1
        fit.coefs *= -1

    return MassDict(kind=kind,
                    k_v=k_v.tolist(),
                    energy=energy,
                    mass_w=mass_w.tolist(),
                    direction_wv=direction_wv.tolist(),
                    max_fit_error=fit_error,
                    fit_data_i=fit.coefs.tolist()))


class NoMinimum(ValueError):
    """Band doesn't have a minimum."""


def fit_band(k_kv: np.ndarray,
             eig_k: np.ndarray) -> tuple[np.ndarray,
                                         float,
                                         np.ndarray,
                                         np.ndarray,
                                         np.ndarray,
                                         PolyFit]:
    dims = k_kv.shape[1]
    npoints = [7, 25, 55][dims - 1]

    k0_v = k_kv[eig_k.argmin()].copy()
    if (k0_v <= k_kv.min(0)).any() or (k0_v >= k_kv.max(0)).any():
        raise NoMinimum('Minimum outside box!')

    k_kv -= k0_v
    k = (k_kv**2).sum(1).argsort()[:npoints]

    if len(k) < npoints:
        raise NoMinimum('Too few points!')

    k_kv = k_kv[k]
    eig_k = eig_k[k]

    try:
        fit = PolyFit(k_kv, eig_k, order=4)
    except np.linalg.LinAlgError:
        raise NoMinimum('Bad minimum!')

    hessian_vv = fit.hessian(np.zeros(dims))
    eval_w = np.linalg.eigvalsh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise NoMinimum('Not a minimum')

    error_k = np.array([fit.value(k_v) - e for k_v, e in zip(k_kv, eig_k)])

    k_v = fit.find_minimum()
    emin = fit.value(k_v)
    hessian_vv = fit.hessian(k_v)
    evals_w, evec_vw = np.linalg.eigh(hessian_vv)
    mass_w = Bohr**2 * Ha / evals_w

    if (mass_w < 0.01).any():
        raise NoMinimum('Unrealistic mass!')

    return k_v + k0_v, emin, mass_w, evec_vw.T, error_k, fit


if __name__ == '__main__':
    main.cli()
