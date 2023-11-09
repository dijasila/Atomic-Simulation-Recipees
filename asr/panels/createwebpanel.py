import warnings
import functools
import numpy as np
from textwrap import wrap

import asr.extra_fluff
from asr.core import ASRResult
from asr.utils.hacks import gs_xcname_from_row
from asr.database.browser import (
    WebPanel,
    create_table, table, matrixtable,
    fig,
    href, dl, code, bold, br, div,
    describe_entry,
    entry_parameter_description,
    make_panel_description)
# from asr.database.browser import href
from asr.extra_fluff import _get_parameter_description, \
    describe_crystaltype_entry, \
    describe_pointgroup_entry, equation, get_dimtypes, get_spg_href, \
    get_summary_table, \
    get_table_row, \
    get_transition_table


# structureinfo
def StructureInfoWebpanel(result, ehullresult, row, key_descriptions):
    PES = href('potential energy surface',
               'https://en.wikipedia.org/wiki/Potential_energy_surface')
    phonon = href('phonon', 'https://en.wikipedia.org/wiki/Phonon')
    stiffnesstensor = href('stiffness tensor',
        'https://en.wikiversity.org/wiki/Elasticity/Constitutive_relations')
    dynstab_description = f"""\
    Dynamically stable materials are stable against small perturbations of
    their structure (atom positions and unit cell shape). The structure
    thus represents a local minimum on the {PES}.

    DS materials are characterised by having only real, non-negative {phonon}
    frequencies and positive definite {stiffnesstensor}.
    """

    spglib = get_spg_href('https://spglib.github.io/spglib/')
    crystal_type = describe_crystaltype_entry(spglib)

    spg_list_link = href(
        'Space group', 'https://en.wikipedia.org/wiki/List_of_space_groups'
    )

    layergroup_link = href(
        'Layer group', 'https://en.wikipedia.org/wiki/Layer_group')

    spacegroup = describe_entry(
        'spacegroup',
        f"{spg_list_link} determined with {spglib}."
        f"The {spg_list_link} determined with {spglib} by stacking the "
        f"monolayer in A-A configuration."
    )

    spgnum = describe_entry(
        'spgnum',
        f"{spg_list_link} number determined with {spglib}."
        f"{spg_list_link} number determined with {spglib} by stacking the "
        f"monolayer in A-A configuration."
    )

    layergroup = describe_entry(
        'layergroup',
        f'{layergroup_link} determined with {spglib}')
    lgnum = describe_entry(
        'lgnum',
        f'{layergroup_link} number determined with {spglib}')

    pointgroup = describe_pointgroup_entry(spglib)

    icsd_link = href('Inorganic Crystal Structure Database (ICSD)',
                     'https://icsd.products.fiz-karlsruhe.de/')

    icsd_id = describe_entry(
        'icsd_id',
        f"ID of a closely related material in the {icsd_link}."
    )

    cod_link = href(
        'Crystallography Open Database (COD)',
        'http://crystallography.net/cod/browse.html'
    )

    cod_id = describe_entry(
        'cod_id',
        f"ID of a closely related material in the {cod_link}."
    )

    # Here we are hacking the "label" out of a row without knowing
    # whether there is a label, or that the "label" recipe exists.

    tablerows = [
        crystal_type, layergroup, lgnum, spacegroup, spgnum, pointgroup,
        icsd_id, cod_id]

    # The table() function is EXTREMELY illogical.
    # I can't get it to work when appending another row
    # to the tablerows list.  Therefore we append rows afterwards.  WTF.
    basictable = table(row, 'Structure info', tablerows, key_descriptions, 2)
    rows = basictable['rows']

    labelresult = row.data.get('results-asr.c2db.labels.json')
    if labelresult is not None:
        tablerow = labelresult.as_formatted_tablerow()
        rows.append(tablerow)

    codid = row.get('cod_id')
    if codid:
        # Monkey patch to make a link
        for tmprow in rows:
            href = ('<a href="http://www.crystallography.net/cod/'
                    + '{id}.html">{id}</a>'.format(id=codid))
            if 'cod_id' in tmprow[0]:
                tmprow[1] = href

    doi = row.get('doi')
    doistring = describe_entry(
        'Reported DOI',
        'DOI of article reporting the synthesis of the material.'
    )
    if doi:
        rows.append([
            doistring,
            '<a href="https://doi.org/{doi}" target="_blank">{doi}'
            '</a>'.format(doi=doi)
        ])

    # XXX There should be a central place defining "summary" panel to take
    # care of stuff that comes not from an individual "recipe" but
    # from multiple ones. Such as stability, or listing multiple band gaps
    # next to each other, etc.
    #
    # For now we stick this in structureinfo but that is god-awful.
    phonon_stability = row.get('dynamic_stability_phonons')
    stiffness_stability = row.get('dynamic_stability_stiffness')

    ehull_table_rows = asr.extra_fluff.ehull_table_rows(row,
                                                        key_descriptions)['rows']

    if phonon_stability is not None and stiffness_stability is not None:
        # XXX This will easily go wrong if 'high'/'low' strings are changed.
        dynamically_stable = (
            phonon_stability == 'high' and stiffness_stability == 'high')

        yesno = ['No', 'Yes'][dynamically_stable]

        dynstab_row = [describe_entry('Dynamically stable', dynstab_description), yesno]
        dynstab_rows = [dynstab_row]
    else:
        dynstab_rows = []

    panel = {'title': 'Summary',
             'columns': [[basictable,
                          {'type': 'table', 'header': ['Stability', ''],
                           'rows': [*ehull_table_rows, *dynstab_rows]}],
                         [{'type': 'atoms'}, {'type': 'cell'}]],
             'sort': -1}

    return [panel]

## Polarizability
def OpticalWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The frequency-dependent polarisability in the long wave length limit (q=0)
    calculated in the random phase approximation (RPA) without spin–orbit
    interactions. For metals a Drude term accounts for intraband transitions. The
    contribution from polar lattice vibrations is added (see infrared
    polarisability) and may be visible at low frequencies.""",
        articles=['C2DB'])
    # breakpoint()
    explanation = 'Optical polarizability along the'
    alphax_el = describe_entry('alphax_el',
                               description=explanation + " x-direction")
    alphay_el = describe_entry('alphay_el',
                               description=explanation + " y-direction")
    alphaz_el = describe_entry('alphaz_el',
                               description=explanation + " z-direction")

    opt = create_table(row=row, header=['Property', 'Value'],
                       keys=[alphax_el, alphay_el, alphaz_el],
                       key_descriptions=key_descriptions, digits=2)

    panel = {'title': describe_entry('Optical polarizability',
                                     panel_description),
             'columns': [[fig('rpa-pol-x.png'), fig('rpa-pol-z.png')],
                         [fig('rpa-pol-y.png'), opt]],
             'plot_descriptions':
                 [{'function': result.polarizability,
                   'filenames': ['rpa-pol-x.png',
                                 'rpa-pol-y.png',
                                 'rpa-pol-z.png']}],
             'subpanel': 'Polarizabilities',
             'sort': 20}

    return [panel]
def PlasmaWebpanel(result, row, key_descriptions):
    from asr.database.browser import table

    if row.get('gap', 1) > 0.01:
        return []

    plasmatable = table(row, 'Property', [
        'plasmafrequency_x', 'plasmafrequency_y'], key_descriptions)

    panel = {'title': 'Optical polarizability',
             'columns': [[], [plasmatable]]}
    return [panel]
def InfraredWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The frequency-dependent polarisability in the infrared (IR) frequency regime
    calculated from a Lorentz oscillator equation involving the optical Gamma-point
    phonons and atomic Born charges. The contribution from electronic interband
    transitions is added, but is essentially constant for frequencies much smaller
    than the direct band gap.
    """,
        articles=[
            href("""\
    M. N. Gjerding et al. Efficient Ab Initio Modeling of Dielectric Screening
    in 2D van der Waals Materials: Including Phonons, Substrates, and Doping,
    J. Phys. Chem. C 124 11609 (2020)""",
                 'https://doi.org/10.1021/acs.jpcc.0c01635'),
        ])
    explanation = 'Static lattice polarizability along the'
    alphax_lat = describe_entry('alphax_lat', description=explanation + " x-direction")
    alphay_lat = describe_entry('alphay_lat', description=explanation + " y-direction")
    alphaz_lat = describe_entry('alphaz_lat', description=explanation + " z-direction")

    explanation = 'Total static polarizability along the'
    alphax = describe_entry('alphax', description=explanation + " x-direction")
    alphay = describe_entry('alphay', description=explanation + " y-direction")
    alphaz = describe_entry('alphaz', description=explanation + " z-direction")

    opt = table(
        row, "Property", [alphax_lat, alphay_lat, alphaz_lat, alphax,
                          alphay, alphaz], key_descriptions
    )

    panel = {
        "title": describe_entry("Infrared polarizability",
                                panel_description),
        "columns": [[fig("infrax.png"), fig("infraz.png")], [fig("infray.png"), opt]],
        "plot_descriptions": [
            {
                "function": result.create_plot,
                "filenames": ["infrax.png", "infray.png", "infraz.png"],
            }
        ],
        "subpanel": 'Polarizabilities',
        "sort": 21,
    }

    return [panel]
# raman
def RamanWebpanel(result, row, key_descriptions):
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
    # Make a table from the phonon modes
    data = row.data.get('results-asr.raman.json')
    if data:
        table = []
        freqs_l = data['freqs_l']
        w_l, rep_l = result.count_deg(freqs_l)
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
                 [{'function': result.raman,
                   'filenames': ['Raman.png']}],
             'sort': 22}

    return [panel]
# SHG
def ShgWebpanel(result, row, key_descriptions):
    # Get the data
    data = row.data.get('results-asr.shg.json')
    if data is None:
        return

    # Make the table
    sym_chi = data.get('symm')
    table = []
    for pol in sorted(sym_chi.keys()):
        relation = sym_chi[pol]
        if pol == 'zero':
            if relation != '':
                pol = 'Others'
                relation = '0=' + relation
            else:
                continue

        if len(relation) == 3:
            relation_new = ''
        else:
            # relation_new = '$'+'$\n$'.join(wrap(relation, 40))+'$'
            relation_new = '\n'.join(wrap(relation, 50))
        table.append((pol, relation_new))
    opt = {'type': 'table',
           'header': ['Element', 'Relations'],
           'rows': table}

    # Make the figure list
    npan = len(sym_chi)
    files = ['shg{}.png'.format(ii + 1) for ii in range(npan)]
    cols = [[fig(f'shg{2 * ii + 1}.png'),
             fig(f'shg{2 * ii + 2}.png')] for ii in range(int(npan / 2))]
    if npan % 2 == 0:
        cols.append([opt, None])
    else:
        cols.append([fig(f'shg{npan}.png'), opt])
    # Transpose the list
    cols = np.array(cols).T.tolist()

    panel = {'title': 'SHG spectrum (RPA)',
             'columns': cols,
             'plot_descriptions':
                 [{'function': result.plot_shg,
                   'filenames': files}],
             'sort': 20}

    return [panel]
# shift
def ShiftWebpanel(result, row, key_descriptions):
    # Get the data
    data = row.data.get('results-asr.shift.json')

    # Make the table
    sym_chi = data.get('symm')
    table = []
    for pol in sorted(sym_chi.keys()):
        relation = sym_chi[pol]
        if pol == 'zero':
            if relation != '':
                pol = 'Others'
                relation = '0=' + relation
            else:
                continue

        if len(relation) == 3:
            relation_new = ''
        else:
            # relation_new = '$'+'$\n$'.join(wrap(relation, 40))+'$'
            relation_new = '\n'.join(wrap(relation, 50))
        table.append((pol, relation_new))
    opt = {'type': 'table',
           'header': ['Element', 'Relations'],
           'rows': table}

    # Make the figure list
    npan = len(sym_chi) - 1
    files = ['shift{}.png'.format(ii + 1) for ii in range(npan)]
    cols = [[fig(f'shift{2 * ii + 1}.png'),
             fig(f'shift{2 * ii + 2}.png')] for ii in range(int(npan / 2))]
    if npan % 2 == 0:
        cols.append([opt, None])
    else:
        cols.append([fig(f'shift{npan}.png'), opt])
    # Transpose the list
    cols = np.array(cols).T.tolist()

    panel = {'title': 'Shift current spectrum (RPA)',
             'columns': cols,
             'plot_descriptions':
                 [{'function': result.plot_shift,
                   'filenames': files}],
             'sort': 20}

    return [panel]
# BSE
def BSEWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The optical absorption calculated from the Bethe–Salpeter Equation
    (BSE). The BSE two-particle Hamiltonian is constructed using the wave functions
    from a DFT calculation with the direct band gap adjusted to match the direct
    band gap from a G0W0 calculation. Spin–orbit interactions are included.  The
    result of the random phase approximation (RPA) with the same direct band gap
    adjustment as used for BSE but without spin–orbit interactions, is also shown.
    """,
        articles=['C2DB'],
    )

    import numpy as np
    from functools import partial

    E_B = table(row, 'Property', ['E_B'], key_descriptions)

    atoms = row.toatoms()
    pbc = atoms.pbc.tolist()
    dim = np.sum(pbc)

    if dim == 2:
        funcx = partial(result.absorption, direction='x')
        funcz = partial(result.absorption, direction='z')

        panel = {'title': describe_entry('Optical absorption (BSE and RPA)',
                                         panel_description),
                 'columns': [[fig('absx.png'), E_B],
                             [fig('absz.png')]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    else:
        funcx = partial(result.absorption, direction='x')
        funcy = partial(result.absorption, direction='y')
        funcz = partial(result.absorption, direction='z')

        panel = {'title': 'Optical absorption (BSE and RPA)',
                 'columns': [[fig('absx.png'), fig('absz.png')],
                             [fig('absy.png'), E_B]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcy,
                                        'filenames': ['absy.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    return [panel]
# gw
def GwWebpanel(result, row, key_descriptions):
    from asr.utils.gw_hse import gw_hse_webpanel
    return gw_hse_webpanel(result, row, key_descriptions, result.get_gw_info(row),
                           sort=16)
# HSE
def HSEWebpanel(result, row, key_descriptions):
    from asr.utils.gw_hse import gw_hse_webpanel
    return gw_hse_webpanel(result, row, key_descriptions,
                           result.get_hse_info(row), sort=12.5)


## Charge
# bader
def BaderWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The Bader charge analysis ascribes a net charge to an atom
    by partitioning the electron density according to its zero-flux surfaces.""",
        articles=[
            href("""W. Tang et al. A grid-based Bader analysis algorithm
    without lattice bias. J. Phys.: Condens. Matter 21, 084204 (2009).""",
                 'https://doi.org/10.1088/0953-8984/21/8/084204')])
    rows = [[str(a), symbol, f'{charge:.2f}']
            for a, (symbol, charge)
            in enumerate(zip(result.sym_a, result.bader_charges))]
    table = {'type': 'table',
             'header': ['Atom index', 'Atom type', 'Charge (|e|)'],
             'rows': rows}

    parameter_description = entry_parameter_description(
        row.data,
        'asr.bader')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Bader charges',
                                     description=title_description),
             'columns': [[table]]}

    return [panel]
# born charges
def BornChargesWebpanel(result, row, key_descriptions):
    reference = """\
    M. N. Gjerding et al. Efficient Ab Initio Modeling of Dielectric Screening
    in 2D van der Waals Materials: Including Phonons, Substrates, and Doping,
    J. Phys. Chem. C 124 11609 (2020)"""
    panel_description = make_panel_description(
        """The Born charge of an atom is defined as the derivative of the static
    macroscopic polarization w.r.t. its displacements u_i (i=x,y,z). The
    polarization in a periodic direction is calculated as an integral over Berry
    phases. The polarization in a non-periodic direction is obtained by direct
    evaluation of the first moment of the electron density. The Born charge is
    obtained as a finite difference of the polarization for displaced atomic
    configurations.  """,
        articles=[
            href(reference, 'https://doi.org/10.1021/acs.jpcc.0c01635')
        ]
    )
    import numpy as np

    def matrixtable(M, digits=2, unit='', skiprow=0, skipcolumn=0):
        table = M.tolist()
        shape = M.shape

        for i in range(skiprow, shape[0]):
            for j in range(skipcolumn, shape[1]):
                value = table[i][j]
                table[i][j] = '{:.{}f}{}'.format(value, digits, unit)
        return table

    columns = [[], []]
    for a, Z_vv in enumerate(
            row.data['results-asr.borncharges.json']['Z_avv']):
        table = np.zeros((4, 4))
        table[1:, 1:] = Z_vv
        rows = matrixtable(table, skiprow=1, skipcolumn=1)
        sym = row.symbols[a]
        rows[0] = [f'Z<sup>{sym}</sup><sub>ij</sub>', 'u<sub>x</sub>',
                   'u<sub>y</sub>', 'u<sub>z</sub>']
        rows[1][0] = 'P<sub>x</sub>'
        rows[2][0] = 'P<sub>y</sub>'
        rows[3][0] = 'P<sub>z</sub>'

        for ir, tmprow in enumerate(rows):
            for ic, item in enumerate(tmprow):
                if ir == 0 or ic == 0:
                    rows[ir][ic] = '<b>' + rows[ir][ic] + '</b>'

        Ztable = dict(
            type='table',
            rows=rows)

        columns[a % 2].append(Ztable)

    panel = {'title': describe_entry('Born charges', panel_description),
             'columns': columns,
             'sort': 17}
    return [panel]
# charge_neutrality
def ChargeNeutralityWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    Equilibrium defect energetics evaluated by solving E<sub>F</sub> self-consistently
    until charge neutrality is achieved.
    """,
        articles=[
            href("""J. Buckeridge, Equilibrium point defect and charge carrier
     concentrations in a meterial determined through calculation of the self-consistent
     Fermi energy, Comp. Phys. Comm. 244 329 (2019)""",
                 'https://doi.org/10.1016/j.cpc.2019.06.017'),
        ],
    )

    unit = result.conc_unit
    unitstring = f"cm<sup>{unit.split('^')[-1]}</sup>"
    panels = []
    for i, scresult in enumerate(result.scresults):
        condition = scresult.condition
        tables = []
        for element in scresult.defect_concentrations:
            conc_table = asr.extra_fluff.get_conc_table(result, element, unitstring)
            tables.append(conc_table)
        scf_overview, scf_summary = asr.extra_fluff.get_overview_tables(scresult,
                                                                        result,
                                                                        unitstring)
        plotname = f'neutrality-{condition}.png'
        panel = WebPanel(
            describe_entry(f'Equilibrium energetics: all defects ({condition})',
                           panel_description),
            columns=[[fig(f'{plotname}'), scf_overview], tables],
            plot_descriptions=[{'function': result.plot_formation_scf,
                                'filenames': [plotname]}],
            sort=25 + i)
        panels.append(panel)

    return panels


## Electronic structure
# bandstructure
def BandstructureWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The band structure with spin–orbit interactions is shown with the
    expectation value of S_i (where i=z for non-magnetic materials and otherwise is
    the magnetic easy axis) indicated by the color code.""",
        articles=['C2DB'],
    )
    bs_png = 'bs.png'
    bs_html = 'bs.html'

    from typing import Tuple, List
    from asr.utils.hacks import gs_xcname_from_row

    def rmxclabel(d: 'Tuple[str, str, str]',
                  xcs: List) -> 'Tuple[str, str, str]':
        def rm(s: str) -> str:
            for xc in xcs:
                s = s.replace('({})'.format(xc), '')
            return s.rstrip()

        return tuple(rm(s) for s in d)

    xcname = gs_xcname_from_row(row)

    panel = {'title': describe_entry(f'Electronic band structure ({xcname})',
                                     panel_description),
             'columns': [
                 [
                     fig(bs_png, link=bs_html),
                 ],
                 [fig('bz-with-gaps.png')]],
             'plot_descriptions': [{'function': result.plot_bs_png,
                                    'filenames': [bs_png]},
                                   {'function': result.plot_bs_html,
                                    'filenames': [bs_html]}],
             'sort': 12}

    return [panel]
# projected bandstructure
scf_projected_bs_filename = 'scf-projected-bs.png'
def ProjBSWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The single-particle band structure and density of states projected onto
    atomic orbitals (s,p,d). Spin–orbit interactions are not included in these
    plots.""",
        articles=[
            'C2DB',
        ],
    )
    xcname = gs_xcname_from_row(row)

    # Projected band structure figure
    parameter_description = entry_parameter_description(
        row.data,
        'asr.bandstructure@calculate')
    dependencies_parameter_descriptions = ''
    for dependency, exclude_keys in zip(
            ['asr.gs@calculate'],
            [set(['txt', 'fixdensity', 'verbose', 'symmetry',
                  'idiotproof', 'maxiter', 'hund', 'random',
                  'experimental', 'basis', 'setups'])]
    ):
        epd = entry_parameter_description(
            row.data,
            dependency,
            exclude_keys=exclude_keys)
        dependencies_parameter_descriptions += f'\n{epd}'
    explanation = ('Orbital projected band structure without spin–orbit coupling\n\n'
                   + parameter_description
                   + dependencies_parameter_descriptions)

    panel = WebPanel(
        title=describe_entry(
            f'Projected band structure and DOS ({xcname})',
            panel_description),
        columns=[[describe_entry(fig(scf_projected_bs_filename, link='empty'),
                                 description=explanation)],
                 [fig('bz-with-gaps.png')]],
        plot_descriptions=[{'function': result.projected_bs_scf,
                            'filenames': [scf_projected_bs_filename]}],
        sort=13.5)

    return [panel]
# dos
def DOSWebpanel(result: ASRResult, row, key_descriptions: dict) -> list:
    panel_description = make_panel_description(
        """Density of States
    """)

    parameter_description = entry_parameter_description(
        row.data,
        'asr.dos')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Density of States',
                                     description=title_description),
             'columns': [[fig('dos.png')]],
             'plot_descriptions':
                 [{'function': result.dos_plot,
                   'filenames': ['dos.png']}]}

    return [panel]
# emasses
def EmassesWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    The effective mass tensor represents the second derivative of the band energy
    w.r.t. wave vector at a band extremum. The effective masses of the valence
    bands (VB) and conduction bands (CB) are obtained as the eigenvalues of the
    mass tensor. The latter is determined by fitting a 2nd order polynomium to the
    band energies on a fine k-point mesh around the band extrema. Spin–orbit
    interactions are included. The fit curve is shown for the highest VB and
    lowest CB. The “parabolicity” of the band is quantified by the
    mean absolute relative error (MARE) of the fit to the band energy in an energy
    range of 25 meV.
    """,
        articles=[
            'C2DB',
        ],
    )
    has_mae = 'results-asr.emasses@validate.json' in row.data
    columns, fnames = result.create_columns_fnames(row)

    electron_dict, hole_dict = result.get_emass_dict_from_row(row, has_mae)

    electron_table = result.custom_table(electron_dict, 'Electron effective '
                                                      'mass',
                                   has_mae)
    hole_table = result.custom_table(hole_dict, 'Hole effective mass', has_mae)
    columns[0].append(electron_table)
    columns[1].append(hole_table)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)

    panel = {'title': describe_entry(f'Effective masses ({xcname})',
                                     panel_description),
             'columns': columns,
             'plot_descriptions':
             [{'function': result.make_the_plots,
               'filenames': fnames
               }],
             'sort': 14}
    return [panel]
# fermisurface
def FermiWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The Fermi surface calculated with spin–orbit interactions. The expectation
    value of S_i (where i=z for non-magnetic materials and otherwise is the
    magnetic easy axis) indicated by the color code.""",
        articles=[
            'C2DB',
        ],
    )
    panel = {'title': describe_entry('Fermi surface', panel_description),
             'columns': [[fig('fermi_surface.png')]],
             'plot_descriptions': [{'function': result.plot_fermi,
                                    'filenames': ['fermi_surface.png']}],
             'sort': 13}

    return [panel]
# pdos
pdos_figfile = 'scf-pdos_nosoc.png'
def PdosWebpanel(result, row, key_descriptions):
    # PDOS figure
    parameter_description = entry_parameter_description(
        row.data,
        'asr.pdos@calculate')
    dependencies_parameter_descriptions = ''
    for dependency, exclude_keys in zip(
            ['asr.gs@calculate'],
            [set(['txt', 'fixdensity', 'verbose', 'symmetry',
                  'idiotproof', 'maxiter', 'hund', 'random',
                  'experimental', 'basis', 'setups'])]
    ):
        epd = entry_parameter_description(
            row.data,
            dependency,
            exclude_keys=exclude_keys)
        dependencies_parameter_descriptions += f'\n{epd}'
    explanation = ('Orbital projected density of states without spin–orbit coupling\n\n'
                   + parameter_description
                   + dependencies_parameter_descriptions)

    xcname = gs_xcname_from_row(row)
    # Projected band structure and DOS panel
    panel = WebPanel(
        title=f'Projected band structure and DOS ({xcname})',
        columns=[[],
                 [describe_entry(fig(pdos_figfile, link='empty'),
                                 description=explanation)]],
        plot_descriptions=[{'function': result.plot_pdos_nosoc,
                            'filenames': [pdos_figfile]}],
        sort=13)

    return [panel]


## Magnetism
# berry
def BerryWebpanel(result, row, key_descriptions):
    olsen_title = ('T. Olsen et al. Discovering two-dimensional topological '
                   'insulators from high-throughput computations. '
                   'Phys. Rev. Mater. 3 024005.')
    olsen_doi = 'https://doi.org/10.1103/PhysRevMaterials.3.024005'

    panel_description = make_panel_description(
        """\
    The spectrum was calculated by diagonalizing the Berry phase matrix
    obtained by parallel transporting the occupied Bloch states along the
    k₀-direction for each value of k₁. The eigenvalues can be interpreted
    as the charge centers of hybrid Wannier functions localised in the
    0-direction and the colours show the expectation values of spin for
    the corresponding Wannier functions. A gapless spectrum is a minimal
    requirement for non-trivial topological invariants.
    """,
        articles=[href(olsen_title, olsen_doi)],
    )
    from asr.utils.hacks import gs_xcname_from_row

    xcname = gs_xcname_from_row(row)
    parameter_description = entry_parameter_description(
        row.data,
        'asr.gs@calculate')
    description = ('Topological invariant characterizing the occupied bands \n\n'
                   + parameter_description)
    datarow = [describe_entry('Band topology', description),
               result.Topology]

    summary = WebPanel(title='Summary',
                       columns=[[{'type': 'table',
                                  'header': ['Basic properties', ''],
                                  'rows': [datarow]}]])

    basicelec = WebPanel(title=f'Basic electronic properties ({xcname})',
                         columns=[[{'type': 'table',
                                    'header': ['Property', ''],
                                    'rows': [datarow]}]],
                         sort=15)

    berry_phases = WebPanel(
        title=describe_entry('Berry phase', panel_description),
        columns=[[fig('berry-phases0.png'),
                  fig('berry-phases0_pi.png')],
                 [fig('berry-phases1.png'),
                  fig('berry-phases2.png')]],
        plot_descriptions=[{'function': result.plot_phases,
                            'filenames': ['berry-phases0.png',
                                          'berry-phases1.png',
                                          'berry-phases2.png',
                                          'berry-phases0_pi.png']}])

    return [summary, basicelec, berry_phases]
# exchange
def ExchangeWebpanel(result, row, key_descriptions):
    from asr.database.browser import (table,
                                      entry_parameter_description,
                                      describe_entry, WebPanel)
    if row.get('magstate', 'NM') == 'NM':
        return []

    parameter_description = entry_parameter_description(
        row.data,
        'asr.exchange@calculate')
    explanation_J = ('The nearest neighbor exchange coupling\n\n'
                     + parameter_description)
    explanation_lam = ('The nearest neighbor isotropic exchange coupling\n\n'
                       + parameter_description)
    explanation_A = ('The single ion anisotropy\n\n'
                     + parameter_description)
    explanation_spin = ('The spin of magnetic atoms\n\n'
                        + parameter_description)
    explanation_N = ('The number of nearest neighbors\n\n'
                     + parameter_description)
    J = describe_entry('J', description=explanation_J)
    lam = describe_entry('lam', description=explanation_lam)
    A = describe_entry('A', description=explanation_A)
    spin = describe_entry('spin', description=explanation_spin)
    N_nn = describe_entry('N_nn', description=explanation_N)

    heisenberg_table = table(row, 'Heisenberg model',
                             [J, lam, A, spin, N_nn],
                             kd=key_descriptions)
    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)
    panel = WebPanel(title=f'Basic magnetic properties ({xcname})',
                     columns=[[heisenberg_table], []],
                     sort=11)
    return [panel]
# hyperfine
def HFWebpanel(result, row, key_description):
    panel_description = make_panel_description(
        """
    Analysis of hyperfine coupling and spin coherence time.
    """,
        articles=[
            href("""G. D. Cheng et al. Optical and spin coherence properties of NV
     center in diamond and 3C-SiC, Comp. Mat. Sc. 154, 60 (2018)""",
                 'https://doi.org/10.1016/j.commatsci.2018.07.039'),
        ],
    )

    hf_results = result.hyperfine
    center = result.center
    if center[0] is None:
        center = [0, 0, 0]

    atoms = row.toatoms()
    args, distances = result.get_atoms_close_to_center(center, atoms)

    hf_table = result.get_hf_table(hf_results, args)
    gyro_table = result.get_gyro_table(row, result)

    hyperfine = WebPanel(describe_entry('Hyperfine (HF) parameters',
                                        panel_description),
                         columns=[[hf_table], [gyro_table]],
                         sort=42)

    return [hyperfine]
# magnetic anisotrophy
# This panel description actually assumes that we also have results for the
# exchange recipe.
def MagAniWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    Heisenberg parameters, magnetic anisotropy and local magnetic
    moments. The Heisenberg parameters were calculated assuming that the
    magnetic energy of atom i can be represented as

      {equation},

    where J is the exchange coupling, B is anisotropic exchange, A is
    single-ion anisotropy and the sums run over nearest neighbours. The
    magnetic anisotropy was obtained from non-selfconsistent spin-orbit
    calculations where the exchange-correlation magnetic field from a
    scalar calculation was aligned with the x, y and z directions.

    """.format(equation=equation()),
        articles=[
            'C2DB',
            href("""D. Torelli et al. High throughput computational screening for 2D
    ferromagnetic materials: the critical role of anisotropy and local
    correlations, 2D Mater. 6 045018 (2019)""",
                 'https://doi.org/10.1088/2053-1583/ab2c43'),
        ],
    )
    if row.get('magstate', 'NM') == 'NM':
        return []

    magtable = table(row, 'Property',
                     ['magstate', 'magmom',
                      'dE_zx', 'dE_zy'], kd=key_descriptions)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)

    panel = {'title':
             describe_entry(
                 f'Basic magnetic properties ({xcname})',
                 panel_description),
             'columns': [[magtable], []],
             'sort': 11}
    return [panel]
# magstate
atomic_mom_threshold = 0.1
def MagStateWebpanel(result, row, key_descriptions):
    """Webpanel for magnetic state."""
    is_magnetic = describe_entry(
        'Magnetic',
        'Is material magnetic?'
        + dl(
            [
                [
                    'Magnetic',
                    code('if max(abs(atomic_magnetic_moments)) > '
                         f'{atomic_mom_threshold}')
                ],
                [
                    'Not magnetic',
                    code('otherwise'),
                ],
            ]
        )
    )

    yesno = ['No', 'Yes'][row.is_magnetic]

    rows = [[is_magnetic, yesno]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Basic properties', ''],
                             'rows': rows}]],
               'sort': 0}

    """
    It makes sense to write the local orbital magnetic moments in the same
    table as the previous local spin magnetic moments; however, orbmag.py was
    added much later than magstate.py, so in order to accomplish this without
    inconvenient changes that may affect other people's projects, we need to
    load the orbmag.py results in this 'hacky' way
    """
    results_orbmag = row.data.get('results-asr.orbmag.json')
    if result.magstate == 'NM':
        return [summary]
    else:
        magmoms_header = ['Atom index', 'Atom type',
                          'Local spin magnetic moment (μ<sub>B</sub>)',
                          'Local orbital magnetic moment (μ<sub>B</sub>)']
        if results_orbmag is None:
            magmoms_rows = [[str(a), symbol, f'{magmom:.3f}', '--']
                            for a, (symbol, magmom)
                            in enumerate(zip(row.get('symbols'),
                                             result.magmoms))]
        else:
            magmoms_rows = [[str(a), symbol, f'{magmom:.3f}', f'{orbmag:.3f}']
                            for a, (symbol, magmom, orbmag)
                            in enumerate(zip(row.get('symbols'),
                                             result.magmoms,
                                             results_orbmag['orbmag_a']))]

        magmoms_table = {'type': 'table',
                         'header': magmoms_header,
                         'rows': magmoms_rows}

        from asr.utils.hacks import gs_xcname_from_row
        xcname = gs_xcname_from_row(row)
        panel = WebPanel(title=f'Basic magnetic properties ({xcname})',
                         columns=[[], [magmoms_table]], sort=11)

        return [summary, panel]
# sj_analyze
def SJAnalyzeWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    Analysis of the thermodynamic stability of the defect using Slater-Janak
     transition state theory.
    """,
        articles=[
            href("""M. Pandey et al. Defect-tolerant monolayer transition metal
    dichalcogenides, Nano Letters, 16 (4) 2234 (2016)""",
                 'https://doi.org/10.1021/acs.nanolett.5b04513'),
        ],
    )
    explained_keys = []
    for key in ['eform']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = key_description
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    defname = row.defect_name
    defstr = f"{defname.split('_')[0]}<sub>{defname.split('_')[1]}</sub>"
    formation_table_sum = get_summary_table(result)
    formation_table = asr.extra_fluff.get_formation_table(result, defstr)
    # defectinfo = row.data.get('asr.defectinfo.json')
    transition_table = get_transition_table(result, defstr)

    panel = WebPanel(
        describe_entry('Formation energies and charge transition levels (Slater-Janak)',
                       panel_description),
        columns=[[describe_entry(fig('sj_transitions.png'),
                                 'Slater-Janak calculated charge transition levels.'),
                  transition_table],
                 [describe_entry(fig('formation.png'),
                                 'Formation energy diagram.'),
                  formation_table]],
        plot_descriptions=[{'function': result.plot_charge_transitions,
                            'filenames': ['sj_transitions.png']},
                           {'function': result.plot_formation_energies,
                            'filenames': ['formation.png']}],
        sort=29)

    summary = {'title': 'Summary',
               'columns': [[formation_table_sum],
                           []],
               'sort': 0}

    return [panel, summary]
# zfs (zero field splitting)
def ZfsWebpanel(result, row, key_description):
    zfs_table = result.get_zfs_table(result)
    zfs = WebPanel('Zero field splitting (ZFS)',
                   columns=[[], [zfs_table]],
                   sort=41)

    return [zfs]


## Convex Hull
# chc
def CHCWebpanel(result, row, key_descriptions):
    from asr.database.browser import fig as asrfig

    fname = 'convexhullcut.png'

    panel = {'title': 'Convex Hull Cut',
             'columns': [[asrfig(fname)]],
             'plot_descriptions':
             [{'function': result.chcut_plot,
               'filenames': [fname]}]}

    return [panel]
# convex_hull
def ConvexHullWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        f'{result.eform_description}\n\n{result.ehull_description}',
        articles=['C2DB'],
    )
    hulltable1 = table(row,
                       'Stability',
                       ['hform', 'ehull'],
                       key_descriptions)
    hulltables = result.convex_hull_tables(row)
    panel = {
        'title': describe_entry(
            'Thermodynamic stability', panel_description),
        'columns': [[fig('convex-hull.png')],
                    [hulltable1] + hulltables],
        'plot_descriptions': [{'function':
                               functools.partial(result.convex_plot,
                                                 thisrow=row),
                               'filenames': ['convex-hull.png']}],
        'sort': 1,
    }

    return [panel]


## Defects
# defect symmetry
def DefectSymmetryWebpanel(result, row, key_descriptions):
    reference = """\
    S. Kaappa et al. Point group symmetry analysis of the electronic structure
    of bare and protected nanocrystals, J. Phys. Chem. A, 122, 43, 8576 (2018)"""

    panel_description = make_panel_description(
        """
    Analysis of defect states localized inside the pristine bandgap (energetics and
     symmetry).
    """,
        articles=[
            href(reference, 'https://doi.org/10.1021/acs.jpca.8b07923'),
        ],
    )

    description = describe_entry('One-electron states', panel_description)
    basictable = asr.extra_fluff.get_summary_table(result, row)

    vbm = result.pristine['vbm']
    cbm = result.pristine['cbm']
    if result.symmetries[0]['best'] is None:
        warnings.warn("no symmetry analysis present for this defect. "
                      "Only plot gapstates!", UserWarning)
        style = 'state'
    else:
        style = 'symmetry'

    state_tables, transition_table = asr.extra_fluff.get_symmetry_tables(
        result.symmetries, vbm, cbm, row, style=style)
    panel = WebPanel(description,
                     columns=[[state_tables[0],
                               fig('ks_gap.png')],
                              [state_tables[1], transition_table]],
                     plot_descriptions=[{'function': result.plot_gapstates,
                                         'filenames': ['ks_gap.png']}],
                     sort=30)

    summary = {'title': 'Summary',
               'columns': [[basictable, transition_table], []],
               'sort': 2}

    return [panel, summary]
# defect info
def DefectInfoWebpanel(result, row, key_descriptions):
    spglib = href('SpgLib', 'https://spglib.github.io/spglib/')
    crystal_type = describe_crystaltype_entry(spglib)

    spg_list_link = href(
        'space group', 'https://en.wikipedia.org/wiki/List_of_space_groups')
    spacegroup = describe_entry(
        'Space group',
        f"The {spg_list_link} is determined with {spglib}.")
    pointgroup = describe_pointgroup_entry(spglib)
    host_hof = describe_entry(
        'Heat of formation',
        result.key_descriptions['host_hof'])
    # XXX get correct XC name
    host_gap_pbe = describe_entry(
        'PBE band gap',
        'PBE band gap of the host crystal [eV].')
    host_gap_hse = describe_entry(
        'HSE band gap',
        'HSE band gap of the host crystal [eV].')
    R_nn = describe_entry(
        'Defect-defect distance',
        result.key_descriptions['R_nn'])

    # extract defect name, charge state, and format it
    defect_name = row.defect_name
    if defect_name != 'pristine':
        defect_name = (f'{defect_name.split("_")[0]}<sub>{defect_name.split("_")[1]}'
                       '</sub>')
        charge_state = row.charge_state
        q = charge_state.split()[-1].split(')')[0]

    # only show results for the concentration if charge neutrality results present
    show_conc = 'results-asr.charge_neutrality.json' in row.data
    if show_conc and defect_name != 'pristine':
        conc_res = row.data['results-asr.charge_neutrality.json']
        conc_row = asr.extra_fluff.get_concentration_row(conc_res, defect_name, q)

    uid = result.host_uid
    uidstring = describe_entry(
        'C2DB link',
        'Link to C2DB entry of the host material.')

    # define overview table with described entries and corresponding results
    lines = [[crystal_type, result.host_crystal],
             [spacegroup, result.host_spacegroup],
             [pointgroup, result.host_pointgroup],
             [host_hof, f'{result.host_hof:.2f} eV/atom'],
             [host_gap_pbe, f'{result.host_gap_pbe:.2f} eV']]
    basictable = table(result, 'Pristine crystal', [])
    basictable['rows'].extend(lines)

    # add additional data to the table if HSE gap, defect-defect distance,
    # concentration, and host uid are present
    if result.host_gap_hse is not None:
        basictable['rows'].extend(
            [[host_gap_hse, f'{result.host_gap_hse:.2f} eV']])
    defecttable = table(result, 'Defect properties', [])
    if result.R_nn is not None:
        defecttable['rows'].extend(
            [[R_nn, f'{result.R_nn:.2f} Å']])
    if show_conc and defect_name != 'pristine':
        defecttable['rows'].extend(conc_row)
    if uid:
        basictable['rows'].extend(
            [[uidstring,
              '<a href="https://cmrdb.fysik.dtu.dk/c2db/row/{uid}"'
              '>{uid}</a>'.format(uid=uid)]])

    panel = {'title': 'Summary',
             'columns': [[basictable, defecttable], []],
             'sort': -1}

    return [panel]
# defect links
def DefectLinksWebpanel(result, row, key_description):
    baselink = 'https://cmrdb.fysik.dtu.dk/qpod/row/'

    # initialize table for charged and neutral systems
    charged_table = table(row, 'Other charge states', [])
    neutral_table = table(row, 'Other defects', [])
    # fill in values for the two tables from the result object
    charged_table = result.extend_table(charged_table, result, 'charged',
                                     baselink)
    neutral_table = result.extend_table(neutral_table, result, 'neutral',
                                     baselink)
    neutral_table = result.extend_table(neutral_table, result, 'pristine',
                                    baselink)

    # define webpanel
    panel = WebPanel('Other defects',
                     columns=[[charged_table], [neutral_table]],
                     sort=45)

    return [panel]


## Structural
# deformationpotentials
def DefPotsWebpanel(result, row, key_descriptions):
    description_text = """\
    The deformation potentials represent the energy shifts of the
    bottom of the conduction band (CB) and the top of the valence band
    (VB) at a given k-point, under an applied strain.

    The two tables at the top show the deformation potentials for the
    valence band (D<sub>VB</sub>) and conduction band (D<sub>CB</sub>)
    at the high-symmetry k-points, subdivided into the different strain
    components. At the bottom of each table are shown the
    deformation potentials at the k-points where the VBM and CBM are found
    (k<sub>VBM</sub> and k<sub>CBM</sub>, respectively).
    Note that the latter may coincide with any of the high-symmetry k-points.
    The table at the bottom shows the band gap deformation potentials.

    All the values shown are calculated with spin-orbit coupling (SOC).
    Values obtained without SOC can be found in the material raw data.
    """
    panel_description = make_panel_description(
        description_text,
        articles=[
            href("""Wiktor, J. and Pasquarello, A., 2016. Absolute deformation potentials
    of two-dimensional materials. Physical Review B, 94(24), p.245411""",
                 "https://doi.org/10.1103/PhysRevB.94.245411")
        ],
    )

    def get_basename(kpt):
        if kpt == 'G':
            return 'Γ'
        elif kpt in ('VBM', 'CBM'):
            return f'k<sub>{kpt}</sub>'
        else:
            return kpt

    description = describe_entry('Deformation potentials', panel_description)
    defpots = result['deformation_potentials_soc'].copy()  # change this back
    # to defpots_soc
    columnlabels = ['xx', 'yy', 'xy']

    dp_gap = defpots.pop('gap')
    dp_list_vb = []
    dp_list_cb = []
    add_to_bottom_vb = []
    add_to_bottom_cb = []
    dp_labels_cb = []
    dp_labels_vb = []

    for kpt in defpots:
        dp_labels = []
        label = get_basename(kpt)
        for band, table, bottom, lab in zip(
                ['VB', 'CB'],
                [dp_list_vb, dp_list_cb],
                [add_to_bottom_vb, add_to_bottom_cb],
                [dp_labels_vb, dp_labels_cb]):
            row = get_table_row(kpt, band, defpots)
            if 'k' in label:
                if band in label:
                    bottom.append((label, row))
                    continue
                else:
                    continue
            lab.append(label)
            table.append(row)

    for label, row in add_to_bottom_vb:
        dp_list_vb.append(row)
        dp_labels_vb.append(label)
    for label, row in add_to_bottom_cb:
        dp_list_cb.append(row)
        dp_labels_cb.append(label)

    dp_labels.append('Band Gap')
    dp_list_gap = [[dp_gap[comp] for comp in ['xx', 'yy', 'xy']]]

    dp_table_vb = matrixtable(
        dp_list_vb,
        digits=2,
        title='D<sub>VB</sub> (eV)',
        columnlabels=columnlabels,
        rowlabels=dp_labels_vb
    )
    dp_table_cb = matrixtable(
        dp_list_cb,
        digits=2,
        title='D<sub>CB</sub> (eV)',
        columnlabels=columnlabels,
        rowlabels=dp_labels_cb
    )
    dp_table_gap = matrixtable(
        dp_list_gap,
        digits=2,
        title='',
        columnlabels=columnlabels,
        rowlabels=['Band Gap']
    )
    panel = WebPanel(
        description,
        columns=[[dp_table_vb, dp_table_gap], [dp_table_cb]],
        sort=4
    )
    return [panel]
# stiffness
def StiffnessWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    The stiffness tensor (C) is a rank-4 tensor that relates the stress of a
    material to the applied strain. In Voigt notation, C is expressed as a NxN
    matrix relating the N independent components of the stress and strain
    tensors. C is calculated as a finite difference of the stress under an applied
    strain with full relaxation of atomic coordinates. A negative eigenvalue of C
    indicates a dynamical instability.
    """,
        articles=['C2DB'],
    )
    import numpy as np

    stiffnessdata = row.data['results-asr.stiffness.json']
    c_ij = stiffnessdata['stiffness_tensor'].copy()
    eigs = stiffnessdata['eigenvalues'].copy()
    nd = np.sum(row.pbc)

    if nd == 2:
        c_ij = np.zeros((4, 4))
        c_ij[1:, 1:] = stiffnessdata['stiffness_tensor']
        ctable = matrixtable(
            stiffnessdata['stiffness_tensor'],
            title='C<sub>ij</sub> (N/m)',
            columnlabels=['xx', 'yy', 'xy'],
            rowlabels=['xx', 'yy', 'xy'])

        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue {ie}', f'{eig.real:.2f} N/m']
                      for ie, eig in enumerate(sorted(eigs,
                                                      key=lambda x: x.real))])
    elif nd == 3:
        eigs *= 1e-9
        c_ij *= 1e-9
        ctable = matrixtable(
            c_ij,
            title='C<sub>ij</sub> (10⁹ N/m²)',
            columnlabels=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'],
            rowlabels=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'])

        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue {ie}', f'{eig.real:.2f} · 10⁹ N/m²']
                      for ie, eig
                      in enumerate(sorted(eigs, key=lambda x: x.real))])
    else:
        ctable = dict(
            type='table',
            rows=[])
        eig = complex(eigs[0])
        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue', f'{eig.real:.2f} * 10⁻¹⁰ N']])

    eigtable = dict(
        type='table',
        rows=eigrows)

    panel = {'title': describe_entry('Stiffness tensor',
                                     description=panel_description),
             'columns': [[ctable], [eigtable]],
             'sort': 2}

    return [panel]
# dimensionality
def DimWebpanel(result, row, key_descriptions):
    dimtable = table(row, 'Dimensionality scores',
                     [f'dim_score_{dimtype}' for dimtype in get_dimtypes()],
                     key_descriptions, 2)
    panel = {'title': 'Dimensionality analysis',
             'columns': [[dimtable], [fig('dimensionality-histogram.png')]]}
    return [panel]


## gs
def GsWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    Electronic properties derived from a ground state density functional theory
    calculation.
    """,
        articles=['C2DB'],
    )

    # for defect systems we don't want to show this panel
    if row.get('defect_name') is not None:
        return []

    parameter_description = _get_parameter_description(row)

    explained_keys = []

    def make_gap_row(name):
        value = result[name]
        description = asr.extra_fluff._explain_bandgap(row, name)
        return [description, f'{value:0.2f} eV']

    gap_row = make_gap_row('gap')
    direct_gap_row = make_gap_row('gap_dir')

    for key in ['dipz', 'evacdiff', 'workfunction', 'dos_at_ef_soc']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = (f'{key_description} '
                           '(Including spin–orbit effects).\n\n'
                           + parameter_description)
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    t = table(result, 'Property',
              explained_keys,
              key_descriptions)

    t['rows'] += [gap_row, direct_gap_row]

    if result.gap > 0:
        if result.get('evac'):
            eref = result.evac
            vbm_title = 'Valence band maximum relative to vacuum level'
            cbm_title = 'Conduction band minimum relative to vacuum level'
            reference_explanation = (
                'the asymptotic value of the '
                'electrostatic potential in the vacuum region')
        else:
            eref = result.efermi
            vbm_title = 'Valence band maximum relative to Fermi level'
            cbm_title = 'Conduction band minimum relative to Fermi level'
            reference_explanation = 'the Fermi level'

        vbm_displayvalue = result.vbm - eref
        cbm_displayvalue = result.cbm - eref
        info = [
            asr.extra_fluff.vbm_or_cbm_row(vbm_title, 'valence band maximum (VBM)',
                                           reference_explanation, vbm_displayvalue),
            asr.extra_fluff.vbm_or_cbm_row(cbm_title, 'conduction band minimum (CBM)',
                                           reference_explanation, cbm_displayvalue)
        ]

        t['rows'].extend(info)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)
    title = f'Basic electronic properties ({xcname})'

    panel = WebPanel(
        title=describe_entry(title, panel_description),
        columns=[[t], [fig('bz-with-gaps.png')]],
        sort=10)

    summary = WebPanel(
        title=describe_entry(
            'Summary',
            description='This panel contains a summary of '
            'basic properties of the material.'),
        columns=[[{
            'type': 'table',
            'header': ['Basic properties', ''],
            'rows': [gap_row],
        }]],
        plot_descriptions=[{'function': result.bz_with_band_extremums,
                            'filenames': ['bz-with-gaps.png']}],
        sort=10)

    return [panel, summary]


## dynamic stability
# phonon
def PhononWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    The Gamma-point phonons of a supercell containing the primitive unit cell
    repeated 2 times along each periodic direction. In the Brillouin zone (BZ) of
    the primitive cell, this yields the phonons at the Gamma-point and
    high-symmetry points at the BZ boundary. A negative eigenvalue of the Hessian
    matrix (the second derivative of the energy w.r.t. to atomic displacements)
    indicates a dynamical instability.
    """,
        articles=['C2DB'],
    )
    phonontable = table(row, 'Property', ['minhessianeig'], key_descriptions)

    panel = {'title': describe_entry('Phonons', panel_description),
             'columns': [[fig('phonon_bs.png')], [phonontable]],
             'plot_descriptions': [{'function': result.plot_bandstructure,
                                    'filenames': ['phonon_bs.png']}],
             'sort': 3}

    return [panel]
# phonopy
def PhonopyWebpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig

    phonontable = table(row, "Property", ["minhessianeig"], key_descriptions)

    panel = {
        "title": "Phonon bandstructure",
        "columns": [[fig("phonon_bs.png")], [phonontable]],
        "plot_descriptions": [
            {"function": result.plot_bandstructure, "filenames": [
                "phonon_bs.png"]}
        ],
        "sort": 3,
    }

    dynstab = row.get("dynamic_stability_level")
    stabilities = {1: "low", 2: "medium", 3: "high"}
    high = "Minimum eigenvalue of Hessian > -0.01 meV/Å² AND elastic const. > 0"
    medium = "Minimum eigenvalue of Hessian > -2 eV/Å² AND elastic const. > 0"
    low = "Minimum eigenvalue of Hessian < -2 eV/Å² OR elastic const. < 0"
    row = [
        "Phonons",
        '<a href="#" data-toggle="tooltip" data-html="true" '
        + 'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'.format(
            low, medium, high, stabilities[dynstab].upper()
        ),
    ]

    summary = {
        "title": "Summary",
        "columns": [
            [
                {
                    "type": "table",
                    "header": ["Stability", "Category"],
                    "rows": [row],
                }
            ]
        ],
    }
    return [panel, summary]


# piezoelectrictensor
def PiezoEleTenWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description("""
    The piezoelectric tensor, c, is a rank-3 tensor relating the macroscopic
    polarization to an applied strain. In Voigt notation, c is expressed as a 3xN
    matrix relating the (x,y,z) components of the polarizability to the N
    independent components of the strain tensor. The polarization in a periodic
    direction is calculated as an integral over Berry phases. The polarization in a
    non-periodic direction is obtained by direct evaluation of the first moment of
    the electron density.
    """)

    piezodata = row.data['results-asr.piezoelectrictensor.json']
    e_vvv = piezodata['eps_vvv']
    e0_vvv = piezodata['eps_clamped_vvv']

    voigt_indices = result.get_voigt_indices(row.pbc)
    voigt_labels = result.get_voigt_labels(row.pbc)

    e_ij = e_vvv[:,
                 voigt_indices[0],
                 voigt_indices[1]]
    e0_ij = e0_vvv[:,
                   voigt_indices[0],
                   voigt_indices[1]]

    etable = matrixtable(e_ij,
                         columnlabels=voigt_labels,
                         rowlabels=['x', 'y', 'z'],
                         title='c<sub>ij</sub> (e/Å<sup>dim-1</sup>)')

    e0table = matrixtable(
        e0_ij,
        columnlabels=voigt_labels,
        rowlabels=['x', 'y', 'z'],
        title='c<sup>clamped</sup><sub>ij</sub> (e/Å<sup>dim-1</sup>)')

    columns = [[etable], [e0table]]

    panel = {'title': describe_entry('Piezoelectric tensor',
                                     panel_description),
             'columns': columns}

    return [panel]
