
# def webpanel(row, key_descriptions):
#     long_names = ['VBM vs. vacuum',
#                   'CBM vs. vacuum',
#                   'Band gap', 'Direct band gap']

#     xcs = ['PBE', 'HSE', 'GLLBSC', 'GW']
#     for xc, xc_name in zip(['', '_hse', '_gllbsc', '_gw'], xcs):
#         for base, s, l in zip(['vbm', 'cbm', 'gap', 'dir_gap'],
#                               ['VBM', 'CBM', '', ''], long_names):
#             key = base + xc
#             add_nosoc += [key]
#             description = '{} ({})'.format(l, xc_name)
#             if s:
#                 value = (s, description, 'eV')
#             else:
#                 value = (description, '', 'eV')
#             key_descriptions[key] = value


group = 'Postprocessing'
