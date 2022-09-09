"""Effective masses - version 117-b."""
from __future__ import annotations

import functools
import sys
from io import StringIO
from math import pi
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from ase.units import Bohr, Ha

from asr.core import ASRResult, command, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  fig, make_panel_description)
from asr.magnetic_anisotropy import get_spin_axis
from asr.utils.eigcalc import EigCalc, GPAWEigenvalueCalculator
from asr.utils.mass_fit import YFunctions


class MassDict(TypedDict):
    kind: str
    k_v: list[float]
    mass_w: list[float]
    direction_wv: list[list[float]]
    energy: float
    max_fit_error: float
    coef_j: list[float]
    lmax: int


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


def _simple(gpwfile: Path | str):
    """For bypassing ASR stuff."""
    from gpaw import GPAW
    eigcalc = GPAWEigenvalueCalculator(GPAW(gpwfile))
    vbm = _main(eigcalc, 'vbm')
    cbm = _main(eigcalc, 'cbm')
    result = EMassesResult.fromdata(vbm_mass=vbm, cbm_mass=cbm)
    row = SimpleNamespace(data={'results-asr.emasses2.json': result})
    mass_plots(row, 'cbm.png', 'vbm.png')


class BadFitError(ValueError):
    """Bad fit to data."""


def kpt2str(kpt_v, cell_cv: np.ndarray) -> str:
    """Pretty-print k-point.

    >>> a = 4.0
    >>> kpt2str([pi / a, 0, 0], np.eye(3) * a)
    '(0.785, 0.000, 0.000) Å^-1 = [0.500, 0.000, 0.000]'
    """
    a, b, c = cell_cv @ kpt_v / (2 * pi)
    x, y, z = kpt_v
    return f'({x:.3f}, {y:.3f}, {z:.3f}) Å^-1 = [{a:.3f}, {b:.3f}, {c:.3f}]'


def find_mass(kpt0_v: np.ndarray,
              kind: str,
              eigcalc: EigCalc,
              kspan: float,
              *,
              maxlevels: int = 5,
              npoints: int = 5,
              max_rel_error: float = 0.01,
              lmax: int = 8,
              fd: StringIO = None) -> MassDict:
    assert kind in {'vbm', 'cbm'}
    log = functools.partial(print, file=fd or sys.stdout)
    K = functools.partial(kpt2str, cell_cv=eigcalc.cell_cv)

    # Create box of k-points centered on kpt0_v:
    shape = tuple(npoints if pbc else 1 for pbc in eigcalc.pbc_c)
    kpt_vx = [np.linspace(kpt - kspan / 2, kpt + kspan / 2, n)
              for kpt, n in zip(kpt0_v, shape)]
    kpt_ijkv = np.empty(shape + (3,))
    kpt_ijkv[..., 0] = kpt_vx[0][:, np.newaxis, np.newaxis]
    kpt_ijkv[..., 1] = kpt_vx[1][:, np.newaxis]
    kpt_ijkv[..., 2] = kpt_vx[2]
    kpt_xv = kpt_ijkv.reshape((-1, 3))
    log(f'{kind.upper()}:')
    log(f'kpts: {K(kpt_xv[0])} ...')
    log(f'      {K(kpt_xv[-1])}')

    eig_x = eigcalc.get_new_band(kind, kpt_xv)
    log(f'eigs: {eig_x.min():.6f} ... {eig_x.max():.6f} [eV]')
    log(f'span: {eig_x.ptp()} [eV]')

    if kind == 'vbm':
        eig_x = -eig_x

    axes = [c for c, size in enumerate(shape) if size > 1]

    try:
        (kpt_v, energy, mass_w, direction_wv,
         max_error, coefs, warping) = fit_band(kpt_xv[:, axes], eig_x, lmax)
    except BadFitError as ex:
        log(ex)
        kspan *= 1.5 / (npoints - 1)
    else:
        if kind == 'vbm':
            extremum = 'maximum'
            energy *= -1
            coefs *= -1
        else:
            extremum = 'minimum'
        kpt0_v[axes] = kpt_v
        log(f'{extremum}: {K(kpt0_v)}, {energy:.6f} [eV]')
        log(f'masses: {mass_w} [a.u.]')
        log(f'warp: {warping * 100} [%]')
        rel_error = max_error / eig_x.ptp()
        log(f'max fit-error: {max_error} [eV] ({rel_error * 100} %)')

        if rel_error <= max_rel_error:
            return MassDict(kind=kind,
                            k_v=kpt_v.tolist(),
                            energy=energy,
                            mass_w=mass_w.tolist(),
                            direction_wv=direction_wv.tolist(),
                            max_fit_error=max_error,
                            coef_j=coefs.tolist(),
                            lmax=lmax)

        log('Error too big')
        kspan *= 0.7 / (npoints - 1)

    if maxlevels == 1:
        raise ValueError

    return find_mass(kpt0_v, kind, eigcalc, kspan,
                     maxlevels=maxlevels - 1,
                     npoints=npoints,
                     max_rel_error=max_rel_error,
                     fd=fd)


def fit_band(k_xv: np.ndarray,
             eig_x: np.ndarray,
             lmax: int = 8) -> tuple[np.ndarray,
                                     float,
                                     np.ndarray,
                                     np.ndarray,
                                     float,
                                     np.ndarray]:
    _, ndims = k_xv.shape
    fit, max_error = YFunctions(ndims, lmax).fit_data(k_xv, eig_x)
    kmin_v = fit.kmin_v
    hessian_vv = fit.hessian()
    eval_w, evec_vw = np.linalg.eigh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise BadFitError('Not a minimum')

    emin = fit.emin
    mass_w = Bohr**2 * Ha / eval_w

    return kmin_v, emin, mass_w, evec_vw.T, max_error, fit.coef_j, fit.warping()


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
        yfuncs = YFunctions(ndims=len(dir_wv), lmax=data['lmax'])
        kmin_v = data['k_v']
        fit = yfuncs.create_fit_from_coefs(data['coef_j'], kmin_v)
        for dir_v in dir_wv:
            k_pv = np.outer(x_p, dir_v) + kmin_v
            y_p = fit.values(k_pv) + offset
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
