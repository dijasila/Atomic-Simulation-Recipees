import numpy as np
from ase.io import read
from ase.io.jsonio import read_json, write_json
from asr.core import command, option


def get_shifts_and_defpots(uid, info):
    info = read_json(info)
    dct = info[str(uid)]
    shift_vb = dct['vbm_GW'] - dct['vbm_PBE']
    shift_cb = dct['cbm_GW'] - dct['cbm_PBE']
    defpots = read_json(dct['defpots'])['kwargs']['data']
    return (shift_vb, shift_cb), defpots


def get_strain_corrections(strain, defpots, subtract: bool=True, soc: bool=False):

    if soc:
        defpots_kb = np.asarray(defpots['deformation_potentials_soc'])
        edges_kb = np.asarray(defpots['edges_soc'])
    else:
        defpots_kb = np.asarray(defpots['deformation_potentials_nosoc'])
        edges_kb = np.asarray(defpots['edges_nosoc'])

    # Let's reshape the strain tensor in a way that is compatible with the deformation potentials
    avg_strain = (strain + strain.T) / 2
    strain_flat = [avg_strain[i] for i in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]]

    newedges = np.zeros_like(edges_kb)
    corrections_kb = newedges.copy()
    for k, defpot_kb in enumerate(defpots_kb):
        for b, band in enumerate(['vb', 'cb']):
            correction = np.dot(strain_flat, defpot_kb[:, b])
            corrections_kb[k, b] = correction
            newedges[k, b] = edges_kb[k, b] + correction

    whereis_new_vbm = np.argmax(newedges[:, 0])
    whereis_new_cbm = np.argmin(newedges[:, 1])

    if not subtract:
        return corrections_kb[whereis_new_vbm, 0], corrections_kb[whereis_new_cbm, 1]
    else:
        return - corrections_kb[whereis_new_vbm, 0], - corrections_kb[whereis_new_cbm, 1]
        

def get_shifts(info, layer_info_file, correct_strain, subtract, soc, only_strain):
    shifts_a, defpots_a = get_shifts_and_defpots(info['uid_a'], layer_info_file)
    shifts_b, defpots_b = get_shifts_and_defpots(info['uid_b'], layer_info_file)
    shifts = np.asarray([*shifts_a, *shifts_b])

    if correct_strain:
        strain_corr_a = get_strain_corrections(info['strain_a'], defpots_a, subtract, soc)
        strain_corr_b = get_strain_corrections(info['strain_b'], defpots_b, subtract, soc)
        corrections = np.asarray([*strain_corr_a, *strain_corr_b])
        if only_strain:
            return corrections
        return shifts + corrections
    return shifts


@command('asr.moire.shifts')
@option ('--info-file')
@option ('--layer-info-file')
@option ('-o', '--output-file')
@option ('--soc', is_flag=True)
@option ('--correct-strain/--dont-correct-strain', is_flag=True)
@option ('--subtract/--add', is_flag=True)
@option ('--only-strain', is_flag=True)
def main(info_file: str = 'bilayer-info.json',
         output_file: str='shifts.json', 
         correct_strain: bool=True,
         layer_info_file: str='/home/niflheim2/steame/moire/utils/layer_info.json',
         soc: bool=False,
         only_strain: bool=False,
         subtract: bool=True):

    info = read_json(info_file)
    for key in info.keys():
        info.update({key: np.asarray(info[key])})

    shifts = get_shifts(info, layer_info_file, correct_strain, subtract, soc, only_strain)
    shifts_dct = {}
    for i, key in enumerate(['shift_v1', 'shift_c1', 'shift_v2', 'shift_c2']):
        shifts_dct.update({key: shifts[i]})

    write_json(output_file, shifts_dct)


if __name__ == '__main__':
    main.cli()
