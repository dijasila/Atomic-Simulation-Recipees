from asr.utils import command, option

@command('asr.hseinterpol')
@option('--kptpath', default=None, type=str)
@option('--npoints', default=400)
def main(kptpath, npoints):
    from asr.hse import bs_interpolate
    results = bs_interpolate(kptpath, npoints)
    return results

if __name__ == '__main__':
    main()


########################################################

#def bs_hse(row, path):
#    from asr.gw import bs_xc
#    bs_xc(row, path, xc='hse', label='HSE')


# def webpanel(row, key_descriptions):
#     from asr.utils.custom import fig, table
#     hse = table('Property',
#                 ['work_function_hse', 'dos_hse', 'gap_hse', 'dir_gap_hse',
#                  'vbm_hse', 'cbm_hse'],
#                  key_descriptions=key_descriptions_noxc)

#     panel = ('Electronic band structure (HSE)',
#              [[fig('hse-bs.png')],
#               [hse]])

#     return panel


group = 'property'
resources = '1:10m'
creates = []
dependencies = ['asr.hse']
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart
