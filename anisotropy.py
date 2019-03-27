
def webpanel(row):
    from asr.custom import table
    if row.magstate != 'NM':
        magtable = table('Property',
                         ['magstate', 'magmom',
                          'maganis_zx', 'maganis_zy', 'dE_NM'])
        panel = ('Magnetic properties', [[magtable], []])
    else:
        panel = []

    return panel


group = 'Property'
