def bs_hse(row, path):
    from asr.gw import bs_xc
    bs_xc(row, path, xc='hse', label='HSE')


def webpanel(row):
    from asr.custom import fig, table
    hse = table('Property',
                ['work_function_hse', 'dos_hse', 'gap_hse', 'dir_gap_hse',
                 'vbm_hse', 'cbm_hse'], key_descriptions=key_descriptions_noxc)

    panel = ('Electronic band structure (HSE)',
             [[fig('hse-bs.png')],
              [hse]])

    return panel


group = 'Property'
