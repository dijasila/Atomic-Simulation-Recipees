from asr.database.browser import (
    WebPanel,
    create_table, table, matrixtable,
    fig,
    href, dl, code, bold, br, div,
    describe_entry,
    entry_parameter_description,
    make_panel_description)

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

## Charge
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
            conc_table = result.get_conc_table(result, element, unitstring)
            tables.append(conc_table)
        scf_overview, scf_summary = result.get_overview_tables(scresult,
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



## Magnetism
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


## Topological


