from asr.database.browser import fig, table, describe_entry


class GWHSEInfo:
    bz_gaps_filename = 'bz-with-gaps.png'

    def __init__(self, result):
        self.result = result

        def _key(key):
            return f'{key}_{self.name}'
        self.gap_key = _key('gap')
        self.gap_dir_key = _key('gap_dir')

    def get(self, name, default=None):
        key = f'{name}_{self.name}'
        return self.result.get(key, default)

    @property
    def gap(self):
        return self.result.get(self.gap_key)

    @property
    def vbm(self):
        return self.get('vbm')

    @property
    def cbm(self):
        return self.get('cbm')


def gw_hse_webpanel(result, context, info, sort):
    key_descriptions = context.descriptions

    ref = context.energy_reference()

    if info.get('gap', 0) > 0.0:
        vbm = info.vbm - ref.value
        cbm = info.cbm - ref.value

        tab = table(result, 'Property',
                    [info.gap_key, info.gap_dir_key],
                    kd=key_descriptions)
        tab['rows'].extend([
            [f'Valence band maximum wrt. {ref.prose_name} ({info.method_name})',
             f'{vbm:.2f} eV'],
            [f'Conduction band minimum wrt. {ref.prose_name} ({info.method_name})',
             f'{cbm:.2f} eV']
        ])

    else:
        tab = table(result, 'Property',
                    [],
                    kd=key_descriptions)

    xcname = context.xcname

    title = f'Electronic band structure ({info.method_name}@{xcname})'
    panel = {'title': describe_entry(title, info.panel_description),
             'columns': [[fig(info.bs_filename)],
                         [fig(info.bz_gaps_filename),
                          tab]],
             'plot_descriptions': [{'function': info.plot_bs,
                                    'filenames': [info.bs_filename]}],
             'sort': sort}

    if info.get('gap'):
        bandgap_entry = describe_entry(
            f'Band gap ({info.method_name})',
            f'The {info.band_gap_adjectives} band gap calculated with '
            f'{info.method_name} including spin–orbit effects.\n\n',
        )
        rows = [[bandgap_entry, f'{info.gap:0.2f} eV']]
        summary = {'title': 'Summary',
                   'columns': [[{'type': 'table',
                                 'header': ['Electronic properties', ''],
                                 'rows': rows}]],
                   'sort': info.summary_sort}
        return [panel, summary]

    return [panel]
