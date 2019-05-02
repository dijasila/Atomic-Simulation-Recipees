
def webpanel(row, key_descriptions):
    from asr.utils.custom import table
    if row.magstate != 'NM':
        magtable = table('Property',
                         ['magstate', 'magmom',
                          'maganis_zx', 'maganis_zy', 'dE_NM'])
        panel = ('Magnetic properties', [[magtable], []])
    else:
        panel = []

    things = ()
    return panel, things


group = 'Property'
dependencies = ['asr.gs']
