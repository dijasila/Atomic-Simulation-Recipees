from asr.utils import command, option

@command('asr.hseinterpol')
@option('--kptpath', default=None, type=str)
@option('--npoints', default=400)
def main(kptpath, npoints):
    from asr.hse import bs_interpolate
    results = bs_interpolate(kptpath, npoints)
    return results

# collect data
"""
def collect_data(atoms):
    # ...
    return kvp, key_descriptions, data 
"""

# from c2db.collect
def hse(kvp, data, atoms, verbose):
    if not op.isfile('hse_bandstructure.npz'):
        return
    if op.isfile('hse_bandstructure3.npz'):
        fname = 'hse_bandstructure3.npz'
    else:
        fname = 'hse_bandstructure.npz'
    dct = dict(np.load(fname))
    if 'epsreal_skn' not in dct:
        warnings.warn('epsreal_skn missing, try and run hseinterpol again')
        return
    print('Collecting HSE bands-structure data')
    evac = kvp.get('evac')
    dct = dict(np.load(fname))
    # without soc first
    data['bs_hse'] = {
        'path': dct['path'],
        'eps_skn': dct['eps_skn'] - evac,
        # 'efermi_nosoc': efermi_nosoc - evac,
        'epsreal_skn': dct['epsreal_skn'] - evac,
        'xkreal': dct['xreal']}

    # then with soc if available
    if 'e_mk' in dct and op.isfile('hse_eigenvalues_soc.npz'):
        e_mk = dct['e_mk']  # band structure
        data['bs_hse'].update(eps_mk=e_mk - evac)

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

if __name__ == '__main__':
    main()
