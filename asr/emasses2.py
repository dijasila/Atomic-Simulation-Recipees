"""Effective masses - version 117-b."""
from __future__ import annotations

import functools
import sys
from io import StringIO
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from ase.units import Bohr, Ha
from scipy.optimize import minimize

from asr.core import ASRResult, command, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  fig, make_panel_description)
from asr.magnetic_anisotropy import get_spin_axis
from asr.utils.eigcalc import EigCalc, GPAWEigenvalueCalculator
from asr.utils.mass_fit import YFit, fit_hessian, fit_values


class MassDict(TypedDict):
    kind: str
    k_v: list[float]
    mass_w: list[float]
    direction_wv: list[list[float]]
    energy: float
    max_fit_error: float
    coef_j: list[float]


@command('asr.emasses2',
         requires=['gs.gpw'],
         dependencies=['asr.gs'])
def main() -> ASRResult:
    """Find effective masses."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw')

    theta, phi = get_spin_axis()
    eigcalc = GPAWEigenvalueCalculator(calc, theta, phi)
    vbm = _main(eigcalc, 'vbm')
    cbm = _main(eigcalc, 'cbm')
    return EMassesResult.fromdata(vbm_mass=vbm, cbm_mass=cbm)


def _main(eigcalc: EigCalc,
          kind: str,
          maxlevels: int = 5) -> MassDict:
    """Find effective masses."""
    kpt_xv, eig_x = eigcalc.get_band(kind)
    if kind == 'vbm':
        x0 = eig_x.argmax()
    else:
        x0 = eig_x.argmin()
    kpt0_v = kpt_xv[x0].copy()

    kdist = ((kpt_xv[1:] - kpt_xv[0])**2).sum(1).min()**0.5
    kspan = 1.5 * kdist

    with open(f'{kind}.log', 'w') as fd:
        massdata = find_mass(kpt0_v, kind, eigcalc, kspan,
                             maxlevels=maxlevels,
                             fd=fd)
    return massdata


class BadFitError(ValueError):
    """Bad fit to data."""


def find_mass(kpt0_v: np.ndarray,
              kind: str,
              eigcalc: EigCalc,
              kspan: float,
              *,
              maxlevels: int = 5,
              npoints: int = 5,
              max_rel_error: float = 0.02,
              fd: StringIO = None) -> MassDict:
    assert kind in {'vbm', 'cbm'}
    log = functools.partial(print, file=fd or sys.stdout)

    # Create box of k-points centered on kpt0_v:
    shape = tuple(npoints if pbc else 1 for pbc in eigcalc.pbc_c)
    kpt_vx = [np.linspace(kpt - kspan / 2, kpt + kspan / 2, n)
              for kpt, n in zip(kpt0_v, shape)]
    kpt_ijkv = np.empty(shape + (3,))
    kpt_ijkv[..., 0] = kpt_vx[0][:, np.newaxis, np.newaxis]
    kpt_ijkv[..., 1] = kpt_vx[1][:, np.newaxis]
    kpt_ijkv[..., 2] = kpt_vx[2]
    kpt_xv = kpt_ijkv.reshape((-1, 3))
    log(f'{kind}:')
    log(f'kpts: {kpt_xv[0]} ...\n      {kpt_xv[-1]} [Ang^-1]')

    eig_x = eigcalc.get_new_band(kind, kpt_xv)
    log(f'eigs: {eig_x.min()} ... {eig_x.max()} [eV]')

    if kind == 'vbm':
        eig_x = -eig_x

    axes = [c for c, size in enumerate(shape) if size > 1]

    try:
        kpt_v, energy, mass_w, direction_wv, error_x, coefs = fit_band(
            kpt_xv[:, axes], eig_x)
    except BadFitError as ex:
        log(ex)
        kspan *= 1.5 / (npoints - 1)
    else:
        if kind == 'vbm':
            energy *= -1
            coefs *= -1
        log(f'extremum: {kpt_v} [Ang^-1], {energy} [eV]')
        log(f'masses: {mass_w}')
        if len(axes) == 3:
            warp = (coefs[7:]**2).sum() / (coefs[1:7]**2).sum()
            log(f'warp: {warp}')
        fit_error = abs(error_x).max()
        log(f'max fit-error: {fit_error} [eV]')
        if fit_error < max_rel_error * eig_x.ptp():
            return MassDict(kind=kind,
                            k_v=kpt_v.tolist(),
                            energy=energy,
                            mass_w=mass_w.tolist(),
                            direction_wv=direction_wv.tolist(),
                            max_fit_error=fit_error,
                            coef_j=coefs.tolist())
        kspan *= 0.7 / (npoints - 1)
        kpt0_v[axes] = kpt_v

    log('Error too big')

    if maxlevels == 1:
        raise ValueError

    return find_mass(kpt0_v, kind, eigcalc, kspan,
                     maxlevels=maxlevels - 1,
                     npoints=npoints,
                     max_rel_error=max_rel_error,
                     fd=fd)


def fit_band(k_xv: np.ndarray,
             eig_x: np.ndarray,
             order: int = 8) -> tuple[np.ndarray,
                                      float,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray]:
    npoints, ndims = k_xv.shape
    if ndims == 1:
        nterms = 3
    elif ndims == 2:
        1 / 0
    else:
        nterms = (order // 2 + 1) * (order + 1) + 4
    if npoints < 1.25 * nterms:
        raise BadFitError('Too few points!')

    kmin_v = k_xv[eig_x.argmin()].copy()
    dk_xv = k_xv - kmin_v
    f = YFit(dk_xv, eig_x, order)
    result = minimize(
        f,
        x0=np.zeros(ndims),
        method='Nelder-Mead')  # seems more robust than the default
    assert result.success
    dkmin_v = result.x
    coefs, error_x = f.fit(dkmin_v)
    hessian_vv = fit_hessian(coefs)
    eval_w, evec_vw = np.linalg.eigh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise BadFitError('Not a minimum')

    emin = coefs[0]
    mass_w = Bohr**2 * Ha / eval_w

    return dkmin_v + kmin_v, emin, mass_w, evec_vw.T, error_x, coefs


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


@prepare_result
class EMassesResult(ASRResult):
    vbm_mass: MassDict
    cbm_mass: MassDict

    key_descriptions = {
        'vbm_mass': 'Mass data for VBM',
        'cbm_mass': 'Mass data for CBM'}

    formats = {'ase_webpanel': webpanel}


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
        coef_j = data['coef_j']
        for dir_v in dir_wv:
            y_p = fit_values(np.outer(x_p, dir_v), coef_j) + offset
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


if __name__ == '__main__':
    main.cli()
