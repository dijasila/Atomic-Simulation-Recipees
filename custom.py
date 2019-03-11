import sys
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import List, Dict, Tuple, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from ase import Atoms
from ase.symbols import string2symbols
from ase.db.row import AtomsRow
from ase.db.summary import create_table, ATOMS, UNITCELL, miscellaneous_section
from ase.dft.band_structure import BandStructure, BandStructurePlot
from ase.dft.kpoints import labels_from_kpts
from ase.units import Ha, Bohr, alpha
from ase.utils import formula_metal

assert sys.version_info >= (3, 4)

title = 'Computational 2D materials database'

plotlyjs = ('<script src="https://cdn.plot.ly/plotly-latest.min.js">' +
            '</script>')
external_libraries = [plotlyjs, ]

key_descriptions = {
    'prototype': ('Prototype', 'Structure prototype', ''),
    'class': ('Class', 'Class of material', ''),
    'ICSD_id': ('ICSD id', 'ICSD id of parent bulk structure', ''),
    'COD_id': ('COD id', 'COD id of parent bulk structure', ''),
    'stoichiometry': ('Stoichiometry', '', ''),
    'folder': ('Folder name', '', ''),
    'has_invsymm': ('Inversion symmetry', '', ''),
    'magstate': ('Magnetic state', 'Magnetic state', ''),
    'hform': ('Heat of formation', '', 'eV/atom'),
    'dE_NM': ('Energy relative to the NM state',
              'Energy relative to the NM state', 'meV/atom'),
    'hsocsplit': ('Hole spin-orbit splitting', '', 'meV'),
    'esocsplit': ('Electron spin-orbit splitting', '', 'meV'),
    'dipz': ('Dipole moment', '', '|e|Ang'),
    'evacmean': ('Mean vacuum level (PBE)', '', 'eV'),
    'evac': ('Vacuum level (no dipole corr) (PBE)', '', 'eV'),
    'evacdiff': ('Vacuum level difference (PBE)', '', 'eV'),
    'evac_gllbsc': ('Vacuum level (no dipole corr) (GLLBSC)', '', 'eV'),
    'excitonmass1': ('Dir. exc. mass 1', 'Direct exciton mass 1 (PBE)',
                     '`m_e`'),
    'excitonmass2': ('Dir. exc. mass 2', 'Direct exciton mass 2 (PBE)',
                     '`m_e`'),
    'emass1': ('Electron mass 1, direction 1', '', '`m_e`'),
    'emass2': ('Electron mass 1, direction 2', '', '`m_e`'),
    'hmass1': ('Hole mass 1, direction 1', '', '`m_e`'),
    'hmass2': ('Hole mass 1, direction 2', '', '`m_e`'),
    'is_2d': ('2D', 'Materials is 2D', ''),
    'is_magnetic': ('Magnetic', 'Material is magnetic', ''),
    'is_dir_gap': ('Direct gap (PBE)', 'Material has direct gap (PBE)', ''),
    'is_metallic': ('Metallic (PBE)', 'Material is metallic', ''),
    'maganis_zx': ('Mag. Anis. (xz)',
                   'Magnetic anisotropy energy (xz-component)',
                   'meV/formula unit'),
    'maganis_zy': ('Mag. Anis. (yz)',
                   'Magnetic anisotropy energy (yz-component)',
                   'meV/formula unit'),
    'spin_orientation': ('Spin orientation',
                         'Magnetic easy crystallographic axis',
                         ''),
    'work_function': ('Work function', '', 'eV'),
    'dosef_nosoc': ('DOS', 'Density of states at Fermi level',
                    '`\\text{eV}^{-1}`'),
    'dosef_soc': ('DOS', 'Density of states at Fermi level',
                  '`\\text{eV}^{-1}`'),
    'nkinds': ('Number of elements', '', ''),
    'dynamic_stability_level': ('Dynamic stability', '', ''),
    'thermodynamic_stability_level': ('Thermodynamic stability', '', ''),
    'c_11': ('Elastic tensor (xx)',
             'Elastic tensor (xx-component)',
             'N/m'),
    'c_22': ('Elastic tensor (yy)',
             'Elastic tensor (yy-component)',
             'N/m'),
    'c_12': ('Elastic tensor (xy)',
             'Elastic tensor (xy-component)',
             'N/m'),
    'speed_of_sound_x': ('Speed of sound (x)', '', 'm/s'),
    'speed_of_sound_y': ('Speed of sound (y)', '', 'm/s'),
    'D_vbm': ('Deformation Pot. (VBM)',
              'Deformation potential at VBM (PBE)',
              'eV'),
    'D_cbm': ('Deformation Pot. (CBM)',
              'Deformation potential at CBM (PBE)',
              'eV'),
    'spacegroup': ('Space group',
                   'Space group',
                   ''),
    'plasmafrequency_x': ('Plasma frequency (x-direction)',
                          'Plasma frequency (x-direction)',
                          '`\\text{eV} \\text{Ang}^{0.5}`'),
    'plasmafrequency_y': ('Plasma frequency (y-direction)',
                          'Plasma frequency (y-direction)',
                          '`\\text{eV} \\text{Ang}^{0.5}`'),
    'alphax': ('Static polarizability (x-direction)',
               'Static polarizability (x-direction)',
               'Ang'),
    'alphay': ('Static polarizability (y-direction)',
               'Static polarizability (y-direction)',
               'Ang'),
    'alphaz': ('Static polarizability (z-direction)',
               'Static polarizability (z-direction)',
               'Ang'),
    'bse_binding': ('Exciton binding energy (BSE)',
                    'Exciton binding energy (BSE)',
                    'eV'),
    'ehull': ('Energy above convex hull', '', 'eV/atom'),
    'minhessianeig': ('Minimum eigenvalue of Hessian', '',
                      '`\\text{eV/Ang}^2`'),
    'monolayer_doi': ('Exp. monolayer reference DOI', '', ''),
    'cell_area': ('Area of unit-cell', '', 'Ang^2'),
    'uid': ('Identifier', '', ''),
    'deltaxc_gllbsc_nosoc': ('Derivative discontinuity (GLLBSC, no SOC)',
                             '', 'eV'),
    'efermi_hse': ('Fermi level (HSE)', '', 'eV'),
    'efermi_gw': ('Fermi level (GW)', '', 'eV'),
    'efermi_hse_nosoc': ('Fermi level (HSE, no SOC)', '', 'eV'),
    'evac_gllbsc_nosoc': ('Vacuum level (no dipole corr) (GLLBSC, no SOC)',
                          '', 'eV'),
    'pbc': ('Periodic boundary conditions', '', ''),
    'smaxinplane': ('Maximum stress in plane', '', '`\\text{eV/Ang}^3`')}

unique_key = 'uid'

add_nosoc = ['D_vbm', 'D_cbm', 'is_metallic', 'is_dir_gap',
             'emass1', 'emass2', 'hmass1', 'hmass2', 'work_function']

long_names = ['VBM vs. vacuum',
              'CBM vs. vacuum',
              'Band gap', 'Direct band gap']

xcs = ['PBE', 'HSE', 'GLLBSC', 'GW']
for xc, xc_name in zip(['', '_hse', '_gllbsc', '_gw'], xcs):
    for base, s, l in zip(['vbm', 'cbm', 'gap', 'dir_gap'],
                          ['VBM', 'CBM', '', ''], long_names):
        key = base + xc
        add_nosoc += [key]
        description = '{} ({})'.format(l, xc_name)
        if s:
            value = (s, description, 'eV')
        else:
            value = (description, '', 'eV')
        key_descriptions[key] = value

for i in range(1, 4):
    for j in range(1, 7):
        key = 'e0_{}{}'.format(i, j)
        name = 'Clamped piezoelectric tensor'
        description = ('{} ({}{})'.format(name, i, j),
                       '{} ({}{}-component)'.format(name, i, j),
                       '`\\text{Ang}^{-1}`')
        key_descriptions[key] = description

        key = 'e_{}{}'.format(i, j)
        name = 'Piezoelectric tensor'
        description = ('{} ({}{})'.format(name, i, j),
                       '{} ({}{}-component)'.format(name, i, j),
                       '`\\text{Ang}^{-1}`')
        key_descriptions[key] = description


def nosoc_update(string):
    if string.endswith(')'):
        return string[:-1] + ', no SOC)'
    else:
        return string + ' (no SOC)'


for key in add_nosoc:
    s, l, units = key_descriptions[key]
    if l:
        key_descriptions[key + "_nosoc"] = (s, nosoc_update(l), units)
    else:
        key_descriptions[key + "_nosoc"] = (nosoc_update(s), l, units)


def rmxclabel(d: 'Tuple[str, str, str]', xcs: List) -> 'Tuple[str, str, str]':
    def rm(s: str) -> str:
        for xc in xcs:
            s = s.replace('({})'.format(xc), '')
        return s.rstrip()
    return tuple(rm(s) for s in d)


default_columns = ['formula', 'prototype', 'magstate',
                   'class', 'spacegroup',
                   'hform', 'gap', 'work_function']

stabilities = {1: 'low', 2: 'medium', 3: 'high'}

special_keys = [
    ('SELECT', 'prototype'),
    ('SELECT', 'class'),
    ('SRANGE', 'dynamic_stability_level', stabilities),
    ('SRANGE', 'thermodynamic_stability_level', stabilities),
    ('SELECT', 'magstate'),
    ('RANGE', 'gap', 'Band gap range [eV]',
     [('PBE', 'gap'),
      ('G0W0@PBE', 'gap_gw'),
      ('GLLBSC', 'gap_gllbsc'),
      ('HSE@PBE', 'gap_hse')])]

params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'savefig.dpi': 200}
plt.rcParams.update(**params)


def val2str(row, key: str, digits=2) -> str:
    value = row.get(key)
    if value is not None:
        if isinstance(value, float):
            value = '{:.{}f}'.format(value, digits)
        elif not isinstance(value, str):
            value = str(value)
    else:
        value = ''
    return value


def fig(filename: str,
        link: str = None) -> 'Dict[str, Any]':
    """Shortcut for figure dict."""
    dct = {'type': 'figure', 'filename': filename}
    if link:
        dct['link'] = link
    return dct


def layout(row: AtomsRow,
           key_descriptions: 'Dict[str, Tuple[str, str, str]]',
           prefix: str) -> 'List[Tuple[str, List[List[Dict[str, Any]]]]]':
    """Page layout."""

    exclude = set()  # Stuff not in miscellaneous section
    xcs = ['PBE', 'GLLBSC', 'HSE', 'GW']
    xcends = ['', '_gllbsc', '_hse', '_gw']
    key_descriptions_noxc = {k: rmxclabel(d, xcs)
                             for k, d in key_descriptions.items()}

    def table(title, keys, digits=2, key_descriptions=key_descriptions):
        exclude.update(keys)
        return create_table(row,
                            [title, 'Value'],
                            keys,
                            key_descriptions,
                            digits)

    if 'c2db-' in prefix:  # make sure links to other rows just works!
        projectname = 'c2db'
    else:
        projectname = 'default'

    # Create simple tables:
    hulltable1 = table('Property',
                       ['hform', 'ehull', 'minhessianeig'])
    hulltable2, hulltable3 = convex_hull_tables(row, projectname)
    phonontable = table('Property',
                        ['c_11', 'c_22', 'c_12', 'bulk_modulus',
                         'minhessianeig'])
    if row.get('gap', 0) > 0.0:
        if row.get('evacdiff', 0) > 0.02:
            pbe = table('Property',
                        ['work_function', 'gap', 'dir_gap',
                         'vbm', 'cbm', 'D_vbm', 'D_cbm', 'dipz', 'evacdiff'],
                        key_descriptions=key_descriptions_noxc)
        else:
            pbe = table('Property',
                        ['work_function', 'gap', 'dir_gap',
                         'vbm', 'cbm', 'D_vbm', 'D_cbm'],
                        key_descriptions=key_descriptions_noxc)
    else:
        if row.get('evacdiff', 0) > 0.02:
            pbe = table('Property',
                        ['work_function', 'dosef_soc', 'gap', 'dir_gap',
                         'vbm', 'cbm', 'D_vbm', 'D_cbm', 'dipz', 'evacdiff'],
                        key_descriptions=key_descriptions_noxc)
        else:
            pbe = table('Property',
                        ['work_function', 'dosef_soc', 'gap', 'dir_gap',
                         'vbm', 'cbm', 'D_vbm', 'D_cbm'],
                        key_descriptions=key_descriptions_noxc)
    gw = table('Property', ['gap_gw', 'dir_gap_gw', 'vbm_gw', 'cbm_gw'],
               key_descriptions=key_descriptions_noxc)
    hse = table('Property',
                ['work_function_hse', 'dos_hse', 'gap_hse', 'dir_gap_hse',
                 'vbm_hse', 'cbm_hse'], key_descriptions=key_descriptions_noxc)
    opt = table('Property', ['alphax', 'alphay', 'alphaz',
                             'plasmafrequency_x', 'plasmafrequency_y'])
    # only show bse if binding energy is there
    if row.get('bse_binding', 0) > 0:
        bse_binding = table('Property',
                            ['bse_binding', 'excitonmass1', 'excitonmass2'])
    else:
        bse_binding = table('Property', [])

    page = [
        ('Basic properties',
         [[basic(row, key_descriptions), UNITCELL],
          [ATOMS]]),
        ('Stability',
         [[fig('convex-hull.png')],
          [hulltable1, hulltable2, hulltable3]]),
        ('Elastic constants and phonons',
         [[fig('phonons.png')], [phonontable]])]
    things = []

    if row.magstate != 'NM':
        magtable = table('Property',
                         ['magstate', 'magmom',
                          'maganis_zx', 'maganis_zy', 'dE_NM'])
        page.append(('Magnetic properties', [[magtable], []]))

    page += [
        ('Electronic band structure (PBE)',
         [[fig('pbe-bs.png', link='pbe-bs.html'),
           fig('bz.png')],
          [fig('pbe-pdos.png', link='empty'), pbe]]),
        ('Effective masses (PBE)',
         [[fig('pbe-bzcut-cb-bs.png'), fig('pbe-bzcut-vb-bs.png')],
          emtables(row)]),
        ('Electronic band structure (HSE)',
         [[fig('hse-bs.png')],
          [hse]]),
        ('Electronic band structure (GW)',
         [[fig('gw-bs.png')],
          [gw]]),
        ('Polarizability (RPA)',
         [[fig('rpa-pol-x.png'), fig('rpa-pol-z.png')],
          [fig('rpa-pol-y.png'), opt]]),
        ('Optical absorption (BSE)',
         [[fig('abs-in.png'), bse_binding],
          [fig('abs-out.png')]])]
    if any(row.get(gap, 0) > 0 for gap in ['gap{}'.format(e)
                                           for e in xcends]):

        def methodtable(prefixes: List[str],
                        xcends: List[str] = xcends,
                        xcs: List[str] = xcs,
                        row=row) -> List[List[str]]:
            exclude.update([prefix + end
                            for end in xcends for prefix in prefixes])
            table = []
            for xc, end in zip(xcs, xcends):
                r = [val2str(row, prefix + end) for prefix in prefixes]
                if not any(r):
                    continue
                table.append([xc] + r)
            return table

        gaptable = dict(
            header=['Method', 'Band gap (eV)', 'Direct band gap (eV)'],
            type='table',
            rows=methodtable(('gap', 'dir_gap')))
        edgetable = dict(
            header=['Method', 'VBM vs vac. (eV)', 'CBM vs vac. (eV)'],
            type='table',
            rows=methodtable(prefixes=('vbm', 'cbm')))

        page += [('Band gaps and -edges (all methods)',
                  [[gaptable, ], [edgetable, ]])]

    from importlib import import_module
    pathlist = Path(__file__).parent.glob('*.py')
    for path in pathlist:
        name = path.with_suffix('').name
        module = import_module('.recipies.' + name, package='mcr')

        try:
            panel, func, figures = module.webpanel(row)
            page += panel
            things.append((func, figures))
        except AttributeError:
            continue

    if 'e_vvv' in row.data:
        def matrixtable(M, digits=2):
            table = M.tolist()
            shape = M.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = table[i][j]
                    table[i][j] = '{:.{}f}'.format(value, digits)
            return table

        e_ij = row.data.e_vvv[:, [0, 1, 2, 1, 0, 0],
                              [0, 1, 2, 2, 2, 1]]
        e0_ij = row.data.e0_vvv[:, [0, 1, 2, 1, 0, 0],
                                [0, 1, 2, 2, 2, 1]]

        etable = dict(
            header=['Piezoelectric tensor', '', ''],
            type='table',
            rows=matrixtable(e_ij))

        e0table = dict(
            header=['Clamped piezoelectric tensor', ''],
            type='table',
            rows=matrixtable(e0_ij))

        panel = [[etable, e0table],
                 [fig('polvsstrain.png'),
                  fig('polvsstrain0.png'),
                  fig('displacementvsstrain.png')]]

        page += [('Piezoelectric tensor', panel)]

    page += [miscellaneous_section(row, key_descriptions, exclude)]

    # List of functions and the figures they create:
    things.extend([(bs_gw, ['gw-bs.png']),
                   (bs_hse, ['hse-bs.png']),
                   (bzcut_pbe, ['pbe-bzcut-cb-bs.png', 'pbe-bzcut-vb-bs.png']),
                   (bs_pbe, ['pbe-bs.png']),
                   (bs_pbe_html, ['pbe-bs.html']),
                   (pdos_pbe, ['pbe-pdos.png']),
                   (bz_soc, ['bz.png']),
                   (absorption, ['abs-in.png', 'abs-out.png']),
                   (polarizability, ['rpa-pol-x.png', 'rpa-pol-y.png',
                                     'rpa-pol-z.png']),
                   (convex_hull, ['convex-hull.png']),
                   (phonons, ['phonons.png']),
                   (polvsstrain, ['polvsstrain.png', 'polvsstrain0.png',
                                  'displacementvsstrain.png'])])

    missing = set()  # missing figures
    for func, filenames in things:
        paths = [Path(prefix + filename) for filename in filenames]
        for path in paths:
            if not path.is_file():
                # Create figure(s) only once:
                func(row, *(str(path) for path in paths))
                for path in paths:
                    if not path.is_file():
                        path.write_text('')  # mark as missing
                break
        for path in paths:
            if path.stat().st_size == 0:
                missing.add(path)

    def ok(block):
        if block is None:
            return False
        if block['type'] == 'table':
            return block['rows']
        if block['type'] != 'figure':
            return True
        if Path(prefix + block['filename']) in missing:
            return False
        return True

    # Remove missing figures from layout:
    final_page = []
    for title, columns in page:
        columns = [[block for block in column if ok(block)]
                   for column in columns]
        if any(columns):
            final_page.append((title, columns))

    return final_page


def add_bs_pbe(row, ax, **kwargs):
    """plot pbe with soc on ax
   """
    c = '0.8'  # light grey for pbe with soc plot
    ls = '-'
    lw = kwargs.get('lw', 1.0)
    d = row.data.bs_pbe
    kpts = d['path']
    e_mk = d['eps_so_mk']
    xcoords, label_xcoords, labels = labels_from_kpts(kpts, row.cell)
    for e_k in e_mk[:-1]:
        ax.plot(xcoords, e_k, color=c, ls=ls, lw=lw, zorder=-2)
    ax.lines[-1].set_label('PBE')
    ef = d['efermi']
    ax.axhline(ef, ls=':', zorder=0, color=c, lw=lw)
    return ax


def pdos_pbe(row, filename='pbe-pdos.png', figsize=(6.4, 4.8),
             fontsize=10, lw=2, loc='best'):
    if 'pdos_pbe' not in row.data:
        return

    def smooth(y, npts=3):
        return np.convolve(y, np.ones(npts) / npts, mode='same')
    dct = row.data.pdos_pbe
    e = dct['energies']
    pdos_sal2 = dct['pdos_sal']
    z_a = set(row.numbers)
    symbols = Atoms(formula_metal(z_a)).get_chemical_symbols()

    def cmp(k):
        s, a, L = k.split(',')
        si = symbols.index(k.split(',')[1])
        li = ['s', 'p', 'd', 'f'].index(L)
        return ('{}{}{}'.format(s, si, li))
    pdos_sal = OrderedDict()
    for k in sorted(pdos_sal2.keys(), key=cmp):
        pdos_sal[k] = pdos_sal2[k]
    colors = {}
    i = 0
    for k in sorted(pdos_sal.keys(), key=cmp):
        if int(k[0]) == 0:
            colors[k[2:]] = 'C{}'.format(i % 10)
            i += 1
    spinpol = False
    for k in pdos_sal.keys():
        if int(k[0]) == 1:
            spinpol = True
            break
    ef = dct['efermi']
    mpl.rcParams['font.size'] = fontsize
    ax = plt.figure(figsize=figsize).add_subplot(111)
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    emin = row.get('vbm', ef) - 3
    emax = row.get('cbm', ef) + 3
    i1, i2 = abs(e - emin).argmin(), abs(e - emax).argmin()
    pdosint_s = defaultdict(float)
    for key in sorted(pdos_sal.keys(), key=cmp):
        pdos = pdos_sal[key]
        spin, spec, lstr = key.split(',')
        spin = int(spin)
        sign = 1 if spin == 0 else -1
        pdosint_s[spin] += np.trapz(y=pdos[i1: i2], x=e[i1: i2])
        if spin == 0:
            label = '{} ({})'.format(spec, lstr)
        else:
            label = None
        ax.plot(smooth(pdos) * sign, e,
                label=label,
                color=colors[key[2:]],
                lw=lw)

    ax.legend(loc=loc)
    ax.axhline(ef, color='k', ls=':')
    ax.set_ylim(emin, emax)
    if spinpol:
        xmax = max(pdosint_s.values())
        ax.set_xlim(-xmax * 0.5, xmax * 0.5)
    else:
        ax.set_xlim(0, pdosint_s[0] * 0.5)

    xlim = ax.get_xlim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.01
    text = ax.annotate(r'$E_\mathrm{F}$', xy=(x0, ef), ha='left', va='bottom',
                       fontsize=fontsize * 1.3)
    text.set_path_effects([path_effects.Stroke(linewidth=3,
                                               foreground='white',
                                               alpha=0.5),
                           path_effects.Normal()])
    ax.set_xlabel('projected dos [states / eV]')
    ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def bs_xc(row, path, xc, **kwargs):
    """xc: 'gw' or 'hse'
    """
    from c2db.bsfitfig import bsfitfig
    lw = kwargs.get('lw', 1)
    if row.data.get('bs_pbe', {}).get('path') is None:
        return
    if 'bs_' + xc not in row.data:
        return
    ax = bsfitfig(row, xc=xc, lw=lw)
    if ax is None:
        return
    label = kwargs.get('label', '?')
    # trying to make the legend label look nice
    for line1 in ax.lines:
        if line1.get_marker() == 'o':
            break
    line0 = ax.lines[0]
    line1, = ax.plot([], [], '-o', c=line0.get_color(),
                     markerfacecolor=line1.get_markerfacecolor(),
                     markeredgecolor=line1.get_markeredgecolor(),
                     markersize=line1.get_markersize(),
                     lw=line0.get_lw())
    line1.set_label(label)
    if 'bs_pbe' in row.data and 'path' in row.data.bs_pbe:
        ax = add_bs_pbe(row, ax, **kwargs)
    ef = row.get('efermi_{}'.format(xc))
    ax.axhline(ef, c='k', ls=':')
    emin = row.get('vbm_' + xc, ef) - 3
    emax = row.get('cbm_' + xc, ef) + 3
    ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    ax.set_ylim(emin, emax)
    ax.set_xlabel('$k$-points')
    leg = ax.legend(loc='upper right')
    leg.get_frame().set_alpha(1)
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    text = ax.annotate(r'$E_\mathrm{F}$', xy=(x0, ef), ha='left', va='bottom',
                       fontsize=13)
    text.set_path_effects([path_effects.Stroke(linewidth=2,
                                               foreground='white',
                                               alpha=0.5),
                           path_effects.Normal()])
    plt.savefig(path)
    plt.close()


def bs_gw(row, path):
    bs_xc(row, path, xc='gw', label='G$_0$W$_0$')


def bs_hse(row, path):
    bs_xc(row, path, xc='hse', label='HSE')


def plot_with_colors(bs, ax=None, emin=-10, emax=5, filename=None,
                     show=None, energies=None, colors=None,
                     ylabel=None, clabel='$s_z$', cmin=-1.0, cmax=1.0,
                     sortcolors=False, loc=None, s=2):
    """Plot band-structure with colors."""

    import matplotlib.pyplot as plt

    if bs.ax is None:
        ax = bs.prepare_plot(ax, emin, emax, ylabel)
    # trying to find vertical lines and putt them in the back

    def vlines2back(lines):
        zmin = min([l.get_zorder() for l in lines])
        for l in lines:
            x = l.get_xdata()
            if len(x) > 0 and np.allclose(x, x[0]):
                l.set_zorder(zmin - 1)
    vlines2back(ax.lines)
    shape = energies.shape
    xcoords = np.vstack([bs.xcoords] * shape[1])
    if sortcolors:
        perm = (-colors).argsort(axis=None)
        energies = energies.ravel()[perm].reshape(shape)
        colors = colors.ravel()[perm].reshape(shape)
        xcoords = xcoords.ravel()[perm].reshape(shape)

    for e_k, c_k, x_k in zip(energies, colors, xcoords):
        things = ax.scatter(x_k, e_k, c=c_k, s=s,
                            vmin=cmin, vmax=cmax)

    cbar = plt.colorbar(things)
    cbar.set_label(clabel)

    bs.finish_plot(filename, show, loc)

    return ax, cbar


def bzcut_pbe(row, pathcb, pathvb, figsize=(6.4, 2.8)):
    from c2db.em import evalmodel
    from ase.dft.kpoints import kpoint_convert
    sortcolors = True
    erange = 0.05  # energy window
    cb = row.get('data', {}).get('effectivemass', {}).get('cb', {})
    vb = row.get('data', {}).get('effectivemass', {}).get('vb', {})

    def getitsorted(keys, bt):
        keys = [k for k in keys if 'spin' in k and 'band' in k]
        return sorted(keys, key=lambda x: int(x.split('_')[1][4:]),
                      reverse=bt == 'vb')

    def get_xkrange(row, erange):
        xkrange = 0.0
        for bt in ['cb', 'vb']:
            xb = row.data.get('effectivemass', {}).get(bt)
            if xb is None:
                continue
            xb0 = xb[getitsorted(xb.keys(), bt)[0]]
            mass_u = abs(xb0['mass_u'])
            xkr = max((2 * mass_u * erange / Ha)**0.5 / Bohr)
            xkrange = max(xkrange, xkr)
        return xkrange

    for bt, xb, path in [('cb', cb, pathcb), ('vb', vb, pathvb)]:
        b_u = xb.get('bzcut_u')
        if b_u is None or b_u == []:
            continue

        xb0 = xb[getitsorted(xb.keys(), bt)[0]]
        mass_u = xb0['mass_u']
        coeff = xb0['c']
        ke_v = xb0['ke_v']
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                                 sharey=True,
                                 gridspec_kw={'width_ratios': [1, 1.25]})
        things = None
        xkrange = get_xkrange(row, erange)
        for u, b in enumerate(b_u):  # loop over directions
            ut = u if bt == 'cb' else abs(u - 1)
            ax = axes[ut]
            e_mk = b['e_dft_km'].T - row.evac
            sz_mk = b['sz_dft_km'].T
            if row.get('has_invsymm', 0) == 1:
                sz_mk[:] = 0.0
            kpts_kc = b['kpts_kc']
            xk, _, _ = labels_from_kpts(kpts=kpts_kc, cell=row.cell)
            xk -= xk[-1] / 2
            # fitted model
            xkmodel = xk.copy()  # xk will be permutated
            kpts_kv = kpoint_convert(row.cell, skpts_kc=kpts_kc)
            kpts_kv *= Bohr
            emodel_k = evalmodel(kpts_kv=kpts_kv, c_p=coeff) * Ha
            emodel_k -= row.evac
            # effective mass fit
            emodel2_k = (xkmodel * Bohr) ** 2 / (2 * mass_u[u]) * Ha
            ecbm = evalmodel(ke_v, coeff) * Ha
            emodel2_k = emodel2_k + ecbm - row.evac
            # dft plot
            shape = e_mk.shape
            x_mk = np.vstack([xk] * shape[0])
            if sortcolors:
                shape = e_mk.shape
                perm = (-sz_mk).argsort(axis=None)
                e_mk = e_mk.ravel()[perm].reshape(shape)
                sz_mk = sz_mk.ravel()[perm].reshape(shape)
                x_mk = x_mk.ravel()[perm].reshape(shape)
            for i, (e_k, sz_k, x_k) in enumerate(zip(e_mk, sz_mk, x_mk)):
                things = ax.scatter(x_k, e_k, c=sz_k, vmin=-1, vmax=1)
            ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
            # ax.plot(xkmodel, emodel_k, c='b', ls='-', label='3rd order')
            sign = np.sign(mass_u[u])
            if (bt == 'cb' and sign > 0) or (bt == 'vb' and sign < 0):
                ax.plot(xkmodel, emodel2_k, c='r', ls='--')
            ax.set_title('Mass {}, direction {}'.format(bt.upper(), ut + 1))
            if bt == 'vb':
                y1 = ecbm - row.evac - erange * 0.75
                y2 = ecbm - row.evac + erange * 0.25
            elif bt == 'cb':
                y1 = ecbm - row.evac - erange * 0.25
                y2 = ecbm - row.evac + erange * 0.75

            ax.set_ylim(y1, y2)
            ax.set_xlim(-xkrange, xkrange)
            ax.set_xlabel(r'$\Delta k$ [1/$\mathrm{\AA}$]')
        if things is not None:
            cbar = fig.colorbar(things, ax=axes[1])
            cbar.set_label(r'$\langle S_z \rangle$')
            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
            cbar.update_ticks()
        fig.tight_layout()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def bs_pbe(row, filename='pbe-bs.png', figsize=(6.4, 4.8),
           fontsize=10, show_legend=True, s=0.5):
    if 'bs_pbe' not in row.data or 'eps_so_mk' not in row.data.bs_pbe:
        return
    d = row.data.bs_pbe
    e_skn = d['eps_skn']
    nspins = e_skn.shape[0]
    e_kn = np.hstack([e_skn[x] for x in range(nspins)])[np.newaxis]
    kpts = d['path']
    ef = d['efermi']
    emin = row.get('vbm', ef) - 3 - ef
    emax = row.get('cbm', ef) + 3 - ef
    mpl.rcParams['font.size'] = fontsize
    bs = BandStructurePlot(BandStructure(row.cell, kpts, e_kn, ef))
    # pbe without soc
    nosoc_style = dict(colors=['0.8'] * e_skn.shape[0],
                       label='PBE no SOC',
                       ls='-',
                       lw=1.0,
                       zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    bs.plot(ax=ax, show=False, emin=emin, emax=emax,
            ylabel=r'$E-E_\mathrm{vac}$ [eV]', **nosoc_style)
    # pbe with soc
    e_mk = d['eps_so_mk']
    sz_mk = d['sz_mk']
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    sdir = row.get('spin_orientation', 'z')
    ax, cbar = plot_with_colors(bs, ax=ax, energies=e_mk, colors=sz_mk,
                                filename=filename, show=False,
                                emin=emin, emax=emax,
                                sortcolors=True, loc='upper right',
                                clabel=r'$\langle S_{} \rangle $'.format(sdir),
                                s=s)

    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.update_ticks()
    csz0 = plt.get_cmap('viridis')(0.5)  # color for sz = 0
    ax.plot([], [], label='PBE', color=csz0)
    ax.set_xlabel('$k$-points')
    plt.legend(loc='upper right')
    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    text = ax.annotate(r'$E_\mathrm{F}$', xy=(x0, ef),
                       ha='left', va='bottom',
                       fontsize=fontsize * 1.3)
    text.set_path_effects([path_effects.Stroke(linewidth=2,
                                               foreground='white',
                                               alpha=0.5),
                           path_effects.Normal()])
    if not show_legend:
        ax.legend_.remove()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def bz_soc(row, fname):
    from c2db.bz import plot_bz
    if 'bs_pbe' in row.data and 'path' in row.data.bs_pbe:
        angle = 0

        plot_bz(row, soc=True, figsize=(5, 5), sfs=1, fname=fname,
                dpi=400, scale=1.5, scalecb=0.8, bbox_to_anchor=(0.5, 0.95),
                angle=angle, scbm=20, svbm=50, lwvbm=1.5)


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


def polarizability(row, fx, fy, fz):
    def xlim():
        return (0, 10)

    def ylims(ws, data, wstart=0.0):
        i = abs(ws - wstart).argmin()
        x = data[i:]
        x1, x2 = x.real, x.imag
        y1 = min(x1.min(), x2.min()) * 1.02
        y2 = max(x1.max(), x2.max()) * 1.02
        return y1, y2

    if 'absorptionspectrum' in row.data:
        data = row.data['absorptionspectrum']
        frequencies = data['frequencies']
        i2 = abs(frequencies - 10.0).argmin()
        frequencies = frequencies[:i2]
        alphax_w = data['alphax_w'][:i2]
        alphay_w = data['alphay_w'][:i2]
        alphaz_w = data['alphaz_w'][:i2]

        ax = plt.figure().add_subplot(111)
        try:
            wpx = row.plasmafrequency_x
            if wpx > 0.01:
                alphaxfull_w = alphax_w - wpx**2 / (2 * np.pi *
                                                    (frequencies + 1e-9)**2)
                ax.plot(frequencies, np.real(alphaxfull_w), '-', c='C1',
                        label='real')
                ax.plot(frequencies, np.real(alphax_w), '--', c='C1',
                        label='real interband')
            else:
                ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
        except AttributeError:
            ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
        ax.plot(frequencies, np.imag(alphax_w), c='C0', label='imag')
        ax.set_title('x-direction')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'polarizability [$\mathrm{\AA}$]')
        ax.set_ylim(ylims(ws=frequencies, data=alphax_w, wstart=0.5))
        ax.legend()
        ax.set_xlim(xlim())
        plt.tight_layout()
        plt.savefig(fx)
        plt.close()

        ax = plt.figure().add_subplot(111)
        try:
            wpy = row.plasmafrequency_y
            if wpy > 0.01:
                alphayfull_w = alphay_w - wpy**2 / (2 * np.pi *
                                                    (frequencies + 1e-9)**2)
                ax.plot(frequencies, np.real(alphayfull_w), '--', c='C1',
                        label='real')
                ax.plot(frequencies, np.real(alphay_w), c='C1',
                        label='real interband')
            else:
                ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')
        except AttributeError:
            ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')
        ax.plot(frequencies, np.imag(alphay_w), c='C0', label='imag')
        ax.set_title('y-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'polarizability [$\mathrm{\AA}$]')
        ax.set_ylim(ylims(ws=frequencies, data=alphax_w, wstart=0.5))
        ax.legend()
        ax.set_xlim(xlim())
        plt.tight_layout()
        plt.savefig(fy)
        plt.close()

        ax = plt.figure().add_subplot(111)
        ax.plot(frequencies, np.real(alphaz_w), c='C1', label='real')
        ax.plot(frequencies, np.imag(alphaz_w), c='C0', label='imag')
        ax.set_title('z-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'polarizability [$\mathrm{\AA}$]')
        ax.set_ylim(ylims(ws=frequencies, data=alphaz_w, wstart=0.5))
        ax.legend()
        ax.set_xlim(xlim())
        plt.tight_layout()
        plt.savefig(fz)
        plt.close()


def emtables(row):
    if row.data.get('effectivemass') is None:
        return [None, None]
    unit = 'm<sub>e</sub>'
    tables = []
    for bt in ['cb', 'vb']:
        dct = row.data.effectivemass.get(bt)
        if dct is None:
            tables.append(None)
            continue
        if bt == 'cb':
            title = 'Electron effective mass'
        else:
            title = 'Hole effective mass'
        keys = [k for k in dct.keys() if 'spin' in k and 'band' in k]
        rows = []
        for i, k in enumerate(keys):
            emdata = dct[k]
            m_u = emdata['mass_u']
            if bt == 'vb':
                m_u = -m_u
            if i == 0:
                desc = '{}'.format(bt.upper())
            else:
                sgn = ' + ' if bt == 'cb' else ' - '
                desc = '{}{}{}'.format(bt.upper(), sgn, i)
            for u, m in enumerate(sorted(m_u, reverse=True)):
                if 0.001 < m < 100:  # masses should be reasonable
                    desc1 = ', direction {}'.format(u + 1)
                    rows.append([desc + desc1,
                                 '{:.2f} {}'.format(m, unit)])
        tables.append({'type': 'table',
                       'header': [title, ''],
                       'rows': rows})
    return tables


def convex_hull(row, fname):
    from ase.phasediagram import PhaseDiagram, parse_formula

    data = row.data.get('chdata')
    if data is None or row.data.get('references') is None:
        return

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
        return

    refs = data['refs']
    pd = PhaseDiagram(refs, verbose=False)

    fig = plt.figure()
    ax = fig.gca()

    if len(count) == 2:
        x, e, names, hull, simplices, xlabel, ylabel = pd.plot2d2()
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '-', color='lightblue')
        ax.plot(x, e, 's', color='C0', label='Bulk')
        dy = e.ptp() / 30
        for a, b, name in zip(x, e, names):
            ax.text(a, b - dy, name, ha='center', va='top')
        A, B = pd.symbols
        ax.set_xlabel('{}$_{{1-x}}${}$_x$'.format(A, B))
        ax.set_ylabel(r'$\Delta H$ [eV/atom]')
        label = '2D'
        ymin = e.min()
        for y, formula, prot, magstate, id, uid in row.data.references:
            count = parse_formula(formula)[0]
            x = count[B] / sum(count.values())
            if id == row.id:
                ax.plot([x], [y], 'rv', label=label)
                ax.plot([x], [y], 'ko', ms=15, fillstyle='none')
            else:
                ax.plot([x], [y], 'v', color='C1', label=label)
            label = None
            ax.text(x + 0.03, y, '{}-{}'.format(prot, magstate))
            ymin = min(ymin, y)
        ax.axis(xmin=-0.1, xmax=1.1, ymin=ymin - 2.5 * dy)
    else:
        x, y, names, hull, simplices = pd.plot2d3()
        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-', color='lightblue')
        ax.plot(x[hull], y[hull], 's', color='C0', label='Bulk (on hull)')
        ax.plot(x[~hull], y[~hull], 's', color='C2', label='Bulk (above hull)')
        for a, b, name in zip(x, y, names):
            ax.text(a - 0.02, b, name, ha='right', va='top')
        A, B, C = pd.symbols
        label = '2D'
        for e, formula, prot, magstate, id, uid in row.data.references:
            count = parse_formula(formula)[0]
            x = count.get(B, 0) / sum(count.values())
            y = count.get(C, 0) / sum(count.values())
            x += y / 2
            y *= 3**0.5 / 2
            if id == row.id:
                ax.plot([x], [y], 'rv', label=label)
                ax.plot([x], [y], 'ko', ms=15, fillstyle='none')
            else:
                ax.plot([x], [y], 'v', color='C1', label=label)
            label = None
        plt.axis('off')

    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def convex_hull_tables(row: AtomsRow,
                       project: str = 'c2db'
                       ) -> 'Tuple[Dict[str, Any], Dict[str, Any]]':
    if row.data.get('references') is None:
        return None, None

    rows = []
    for e, formula, prot, magstate, id, uid in sorted(row.data.references,
                                                      reverse=True):
        name = '{} ({}-{})'.format(formula, prot, magstate)
        if id != row.id:
            name = '<a href="/{}/row/{}">{}</a>'.format(project, uid, name)
        rows.append([name, '{:.3f} eV/atom'.format(e)])

    refs = row.data.get('chdata')['refs']
    bulkrows = []
    for formula, e in refs:
        e /= len(string2symbols(formula))
        link = '<a href="/oqmd12/row/{formula}">{formula}</a>'.format(
            formula=formula)
        bulkrows.append([link, '{:.3f} eV/atom'.format(e)])

    return ({'type': 'table',
             'header': ['Monolayer formation energies', ''],
            'rows': rows},
            {'type': 'table',
             'header': ['Bulk formation energies', ''],
             'rows': bulkrows})


def phonons(row, fname):
    freqs = row.data.get('phonon_frequencies_3d')
    if freqs is None:
        return

    gamma = freqs[0]
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()

    x0 = -0.0005  # eV
    for x, color in [(gamma[gamma < x0], 'r'),
                     (gamma[gamma >= x0], 'b')]:
        if len(x) > 0:
            markerline, _, _ = ax.stem(x * 1000, np.ones_like(x), bottom=-1,
                                       markerfmt=color + 'o',
                                       linefmt=color + '-')
            plt.setp(markerline, alpha=0.4)
    ax.set_xlabel(r'phonon frequency at $\Gamma$ [meV]')
    ax.axis(ymin=0.0, ymax=1.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# @creates('stability.html', 'convex-hull.csv')
def plotly_stability(row):
    """Some work on making an interactive stability figure.
    Still a work in progress.
    """

    import plotly
    import plotly.graph_objs as go
    from ase.phasediagram import PhaseDiagram, parse_formula

    data = row.data.get('chdata')
    if data is None or row.data.get('references') is None:
        print(data, row.data.get('references'))
        return

    count = row.count_atoms()
    if len(count) > 3:
        return

    refs = data['refs']
    pd = PhaseDiagram(refs, verbose=False)

    if len(count) == 2:
        x, e, names, hull, simplices, xlabel, ylabel = pd.plot2d2()
        A, B = pd.symbols
        data = []

        for i, j in simplices:
            data.append(go.Scatter(x=x[[i, j]], y=e[[i, j]], showlegend=False,
                                   hoverinfo='skip',
                                   line={'color': ('lightblue')}))

        htmlnames = []

        for name in names:
            name = name.replace('$_{', '<sub>').replace('}$', '</sub>')
            htmlnames.append(name)

        data.append(go.Scatter(x=x, y=e, text=htmlnames,
                               hoverinfo='x+y+text+name',
                               name='Bulk',
                               marker={'color': 'green',
                                       'symbol': 'square',
                                       'size': "20"},
                               mode='markers'))
        xlabel = '{}<sub>1-x</sub>{}<sub>x</sub>'.format(A, B)
        ylabel = r'$\Delta H$ [ev/atom]'
        layout = {'xaxis': {'title': xlabel},
                  'yaxis': {'title': ylabel}}

        for y, formula, prot, magstate, id in row.data.references:
            count = parse_formula(formula)[0]
            x = count[B] / sum(count.values())
            if id == row.id:
                data.append(go.Scatter(x=[x], y=[y], text=[formula],
                                       mode='markers',
                                       showlegend=False,
                                       hoverinfo='x+y+text',
                                       marker={'symbol': 'triangle-down',
                                               'color': 'red',
                                               'size': "20"}))
                data.append(go.Scatter(x=[x], y=[y], mode='markers',
                                       showlegend=False,
                                       hoverinfo='skip',
                                       marker={'symbol': 'circle-open',
                                               'color': 'red',
                                               'size': "25"}))

                # ax.plot([x], [y], 'ko', ms=15, fillstyle='none')
            else:
                data.append(go.Scatter(x=[x], y=[y], text=[formula],
                                       mode='markers',
                                       hoverinfo='x+y+text',
                                       name='2D',
                                       marker={'symbol': 'triangle-down',
                                               'color': 'C1',
                                               'size': "20"}))
                # ax.plot([x], [y], 'v', color='C1', label=label)
            # label = None
            # ax.text(x + 0.03, y, '{}-{}'.format(prot, magstate))
            # ymin = min(ymin, y)

    fig = {'data': data, 'layout': layout}
    fig['layout']['margin'] = {'t': 25, 'b': 50, 'l': 50, 'r': 50}
    html = plotly.offline.plot(fig,
                               include_plotlyjs=False,
                               output_type='div')

    with open('stability.html', 'w') as fd:
        fd.write(html)

    with open('convex-hull.csv', 'w') as f:
        f.write('# Formation energies\n')
        for e, formula, prot, magstate, id in sorted(row.data.references,
                                                     reverse=True):
            name = '{} ({}-{})'.format(formula, prot, magstate)
            if id != row.id:
                name = '<a href="/id/{}?project=c2dm">{}</a>'.format(id, name)
            f.write('{}, {:.3f}, eV/atom\n'.format(name, e))


# @creates('pbe-bs.html')
def bs_pbe_html(row, filename='pbe-bs.html', figsize=(6.4, 4.8),
                fontsize=10, show_legend=True, s=2):
    if 'bs_pbe' not in row.data or 'eps_so_mk' not in row.data.bs_pbe:
        return

    import plotly
    import plotly.graph_objs as go

    traces = []
    d = row.data.bs_pbe
    e_skn = d['eps_skn']
    kpts = d['path']
    ef = d['efermi']
    emin = row.get('vbm', ef) - 5
    emax = row.get('cbm', ef) + 5
    shape = e_skn.shape

    from ase.dft.kpoints import labels_from_kpts
    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, row.cell)
    xcoords = np.vstack([xcoords] * shape[0] * shape[2])
    # colors_s = plt.get_cmap('viridis')([0, 1])  # color for sz = 0
    e_kn = np.hstack([e_skn[x] for x in range(shape[0])])
    trace = go.Scattergl(x=xcoords.ravel(), y=e_kn.T.ravel(),
                         mode='markers',
                         name='PBE no SOC',
                         showlegend=True,
                         marker=dict(
                             size=4,
                             color='#999999'))
    traces.append(trace)

    d = row.data.bs_pbe
    e_mk = d['eps_so_mk']
    kpts = d['path']
    ef = d['efermi']
    sz_mk = d['sz_mk']
    emin = row.get('vbm', ef) - 5
    emax = row.get('cbm', ef) + 5

    from ase.dft.kpoints import labels_from_kpts
    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, row.cell)

    shape = e_mk.shape
    perm = (sz_mk).argsort(axis=None)
    e_mk = e_mk.ravel()[perm].reshape(shape)
    sz_mk = sz_mk.ravel()[perm].reshape(shape)
    xcoords = np.vstack([xcoords] * shape[0])
    xcoords = xcoords.ravel()[perm].reshape(shape)

    # Unicode for <S_z>
    sdir = row.get('spin_orientation', 'z')
    cbtitle = '&#x3008; <i><b>S</b></i><sub>{}</sub> &#x3009;'.format(sdir)
    trace = go.Scattergl(x=xcoords.ravel(), y=e_mk.ravel(),
                         mode='markers',
                         name='PBE',
                         showlegend=True,
                         marker=dict(
                             size=4,
                             color=sz_mk.ravel(),
                             colorscale='Viridis',
                             showscale=True,
                             colorbar=dict(tickmode='array',
                                           tickvals=[-1, 0, 1],
                                           ticktext=['-1', '0', '1'],
                                           title=cbtitle,
                                           titleside='right')))
    traces.append(trace)

    linetrace = go.Scatter(x=[np.min(xcoords), np.max(xcoords)],
                           y=[ef, ef],
                           mode='lines',
                           line=dict(color=('rgb(0, 0, 0)'),
                                     width=2,
                                     dash='dash'),
                           name='Fermi level')
    traces.append(linetrace)

    def pretty(kpt):
        if kpt == 'G':
            kpt = '&#x393;'  # Gamma in unicode
        elif len(kpt) == 2:
            kpt = kpt[0] + '$_' + kpt[1] + '$'
        return kpt

    labels = [pretty(name) for name in orig_labels]
    i = 1
    while i < len(labels):
        if label_xcoords[i - 1] == label_xcoords[i]:
            labels[i - 1] = labels[i - 1][:-1] + ',' + labels[i][1:]
            labels[i] = ''
        i += 1

    bandxaxis = go.layout.XAxis(
        title="k-points",
        range=[0, np.max(xcoords)],
        showgrid=True,
        showline=True,
        ticks="",
        showticklabels=True,
        mirror=True,
        linewidth=2,
        ticktext=labels,
        tickvals=label_xcoords,
    )

    bandyaxis = go.layout.YAxis(
        title="<i>E - E</i><sub>vac</sub> [eV]",
        range=[emin, emax],
        showgrid=True,
        showline=True,
        zeroline=False,
        mirror="ticks",
        ticks="inside",
        linewidth=2,
        tickwidth=2,
        zerolinewidth=2,
    )

    bandlayout = go.Layout(
        xaxis=bandxaxis,
        yaxis=bandyaxis,
        legend=dict(x=0, y=1),
        hovermode='closest',
        margin=dict(t=40, r=100),
        font=dict(size=18))

    fig = {'data': traces, 'layout': bandlayout}
    # fig['layout']['margin'] = {'t': 40, 'r': 100}
    # fig['layout']['hovermode'] = 'closest'
    # fig['layout']['legend'] =

    plot_html = plotly.offline.plot(fig,
                                    include_plotlyjs=False,
                                    output_type='div')
    # plot_html = ''.join(['<div style="width: 1000px;',
    #                      'height=1000px;">',
    #                      plot_html,
    #                      '</div>'])

    inds = []
    for i, c in enumerate(plot_html):
        if c == '"':
            inds.append(i)
    plotdivid = plot_html[inds[0] + 1:inds[1]]

    resize_script = (
        ''
        '<script type="text/javascript">'
        'window.addEventListener("resize", function(){{'
        'Plotly.Plots.resize(document.getElementById("{id}"));}});'
        '</script>'
    ).format(id=plotdivid)

    # Insert plotly.js
    plotlyjs = ('<script src="https://cdn.plot.ly/plotly-latest.min.js">' +
                '</script>')

    html = ''.join(['<html>',
                    '<head><meta charset="utf-8" /></head>',
                    '<body>',
                    plotlyjs,
                    plot_html,
                    resize_script,
                    '</body>',
                    '</html>'])

    with open(filename, 'w') as fd:
        fd.write(html)


def basic(row, key_descriptions):
    table = create_table(row,
                         ['Property', 'Value'],
                         ['prototype', 'class', 'spacegroup', 'gap',
                          'magstate', 'ICSD_id'],
                         key_descriptions,
                         2)
    rows = table['rows']
    codid = row.get('COD_id')
    if codid:
        href = ('<a href="http://www.crystallography.net/cod/' +
                '{id}.html">{id}</a>'.format(id=codid))
        rows.append([key_descriptions['COD_id'][1], href])
    dynstab = row.get('dynamic_stability_level')
    if dynstab:
        high = 'Min. Hessian eig. > -0.01 meV/Ang^2 AND elastic const. > 0'
        medium = 'Min. Hessian eig. > -2 eV/Ang^2 AND elastic const. > 0'
        low = 'Min. Hessian eig.  < -2 eV/Ang^2 OR elastic const. < 0'
        rows.append(
            ['Dynamic stability',
             '<a href="#" data-toggle="tooltip" data-html="true" ' +
             'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'
             .format(low, medium, high, stabilities[dynstab].upper())])

    thermostab = row.get('thermodynamic_stability_level')
    if thermostab:
        high = 'Heat of formation < convex hull + 0.2 eV/atom'
        medium = 'Heat of formation < 0.2 eV/atom'
        low = 'Heat of formation > 0.2 eV/atom'
        rows.append(
            ['Thermodynamic stability',
             '<a href="#" data-toggle="tooltip" data-html="true" ' +
             'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'
             .format(low, medium, high, stabilities[thermostab].upper())])

    doi = row.get('monolayer_doi')
    if doi:
        rows.append(
            ['Monolayer DOI',
             '<a href="https://doi.org/{doi}" target="_blank">{doi}'
             '</a>'.format(doi=doi)])

    return table


def polvsstrain(row, f1, f0, f2):

    if 'piezodata' not in row.data:
        return

    piezodata = row.data['piezodata']

    deltas_d = piezodata['deltas_d']
    P1 = piezodata['true']
    P0 = piezodata['clamped']

    eqspos_ac = piezodata['eqspos_ac']
    spos_dvvac = piezodata['spos_dvvac'] - eqspos_ac
    spos_dvvac -= np.round(spos_dvvac)
    
    pos_dvvav = np.dot(spos_dvvac, row.cell)

    plt.figure()
    na = pos_dvvav.shape[3]
    for i in range(3):
        for j in range(3):
            if j < i:
                continue
                
            for a in range(na):
                for v in range(3):
                    pos_d = pos_dvvav[:, i, j, a, v]
                    if np.max(pos_d) < np.max(pos_dvvav) / 20:
                        continue
                    plt.plot(deltas_d, pos_d, '-o',
                             label='{}{}{}{}'.format(i, j, a, v))

    plt.xlabel(r'eps (\%)')
    plt.ylabel('Displacement (Å)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f2)
    plt.close()

    for dat, filename, epsstring in [[P1, f1, 'e'], [P0, f0, 'e0']]:
        plt.figure()
        P_dvvv = dat

        P_dvv = P_dvvv[:, :,
                       [0, 1, 2, 1, 0, 0],
                       [0, 1, 2, 2, 2, 1]]

        Pm_vv = np.mean(P_dvv, axis=0)
        P_dvv -= Pm_vv

        Pmax = np.max(np.abs(P_dvv))
        for i in range(3):
            for j in range(6):
                P_d = P_dvv[:, i, j]
                if np.max(np.abs(P_d)) < Pmax / 20:
                    continue
                eps = row['{}_{}{}'.format(epsstring, i + 1, j + 1)]
                plt.plot(deltas_d, eps * deltas_d, 'k--')
                plt.plot(deltas_d, P_d, '-o',
                         label='{}{}'.format(i, j))

        plt.xlabel(r'eps (\%)')
        plt.ylabel('Pol')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
