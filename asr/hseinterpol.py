from asr.utils import command, option, read_json
import os

@command('asr.hseinterpol')
@option('--kptpath', default=None, type=str)
@option('--npoints', default=400)
def main(kptpath, npoints):
    from asr.hse import bs_interpolate
    results = bs_interpolate(kptpath, npoints)
    return results

def collect_data(atoms):
    """
    Collect results obtained with 2 methods:
    1) ase.dft.kpoints.monkhorst_pack_interpolate
    2) interpolation scheme interpolate_bandlines2(calc, path, e_skn=None)

    XXX
    WARNING: the latter should be ok for 2D, but don't work well for 3D
    we need a better interpolation scheme to work with 3D structures
    """

    kvp = {}
    key_descriptions = {}
    data = {}

    evac = 0.0 # XXX where do I find evac?
    #evac = kvp.get('evac')

    if not os.path.isfile('results_hseinterpol.json'):
        return kvp, key_descriptions, data

    results = read_json('results_hseinterpol.json')

    """
    1) Results obtained with ase.dft.kpoints.monkhorst_pack_interpolate
    """
    dct = results['hse_bandstructure']
     # without soc first
    data['hse_interpol'] = {
        'path': dct['path'],
        'eps_skn': dct['eps_skn'] - evac}
    # then with soc if available
    if 'e_mk' in dct:
        e_mk = dct['e_mk']  # band structure
        s_mk = dct['s_mk']
        data['hse_interpol'].update(eps_mk=e_mk - evac, s_mk=s_mk)

    """
    2) Results from interpolate_bandlines2(calc, path, e_skn=None)
    XXX Warning: not suitable for 3D structures!
    """
    dct = results['hse_bandstructure3']
    if 'epsreal_skn' not in dct:
        warnings.warn('epsreal_skn missing, try and run hseinterpol again')
        return kvp, key_descriptions, data
 
    # without soc first
    data['hse_interpol3'] = {
        'path': dct['path'],
        'eps_skn': dct['eps_skn'] - evac,
        'epsreal_skn': dct['epsreal_skn'] - evac,
        'xkreal': dct['xreal']}
    # then with soc if available
    if 'e_mk' in dct:
        e_mk = dct['e_mk']  # band structure
        data['hse_interpol3'].update(eps_mk=e_mk - evac)

    return kvp, key_descriptions, data

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
