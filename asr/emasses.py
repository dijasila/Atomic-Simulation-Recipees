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


# def webpanel(row, key_descriptions):

#     from asr.utils.custom import fig
#     add_nosoc = ['D_vbm', 'D_cbm', 'is_metallic', 'is_dir_gap',
#                  'emass1', 'emass2', 'hmass1', 'hmass2', 'work_function']

#     def nosoc_update(string):
#         if string.endswith(')'):
#             return string[:-1] + ', no SOC)'
#         else:
#             return string + ' (no SOC)'

#     for key in add_nosoc:
#         s, l, units = key_descriptions[key]
#         if l:
#             key_descriptions[key + "_nosoc"] = (s, nosoc_update(l), units)
#         else:
#             key_descriptions[key + "_nosoc"] = (nosoc_update(s), l, units)

#     panel = ('Effective masses (PBE)',
#              [[fig('pbe-bzcut-cb-bs.png'), fig('pbe-bzcut-vb-bs.png')],
#               emtables(row)])

#     return panel

            
group = 'Property'
