from asr.database.browser import (
    fig, table, describe_entry, make_panel_description)


def gw_hse_webpanel(result, row, key_descriptions, info):
    if row.get('gap_hse', 0) > 0.0:
        hse = table(row, 'Property',
                    ['gap_hse', 'gap_dir_hse'],
                    kd=key_descriptions)

        if row.get('evac'):
            hse['rows'].extend(
                [['Valence band maximum wrt. vacuum level (HSE06)',
                  f'{row.vbm_hse - row.evac:.2f} eV'],
                 ['Conduction band minimum wrt. vacuum level (HSE06)',
                  f'{row.cbm_hse - row.evac:.2f} eV']])
        else:
            hse['rows'].extend(
                [['Valence band maximum wrt. Fermi level (HSE06)',
                  f'{row.vbm_hse - row.efermi:.2f} eV'],
                 ['Conduction band minimum wrt. Fermi level (HSE06)',
                  f'{row.cbm_hse - row.efermi:.2f} eV']])
    else:
        hse = table(row, 'Property',
                    [],
                    kd=key_descriptions)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)

    title = f'Electronic band structure (HSE06@{xcname})'
    panel = {'title': describe_entry(title, info.panel_description),
             'columns': [[fig('hse-bs.png')],
                         [fig('bz-with-gaps.png'), hse]],
             'plot_descriptions': [{'function': info.plot_bs,
                                    'filenames': ['hse-bs.png']}],
             'sort': 15}

    if row.get('gap_hse'):

        bandgaphse = describe_entry(
            'Band gap (HSE)',
            'The electronic single-particle band gap calculated with '
            'HSE including spin–orbit effects.\n\n',
        )
        rows = [[bandgaphse, f'{row.gap_hse:0.2f} eV']]
        summary = {'title': 'Summary',
                   'columns': [[{'type': 'table',
                                 'header': ['Electronic properties', ''],
                                 'rows': rows}]],
                   'sort': 11}
        return [panel, summary]

    return [panel]
