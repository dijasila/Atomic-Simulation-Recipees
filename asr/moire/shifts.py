from asr.core import command, option
from ase.io.jsonio import read_json, write_json
#from asr.moire.makemoire import get_parameters, get_atoms_and_stiffness, make_bilayer
import numpy as np


def get_shifts_and_defpots(uid, info):
    info = read_json(info)
    dct = info[str(uid)]
    shift_vb = dct['vbm_GW'] - dct['vbm_PBE']
    shift_cb = dct['cbm_GW'] - dct['cbm_PBE']
    defpot_vb = dct['defpot_vbm']
    defpot_cb = dct['defpot_cbm']
    return (shift_vb, shift_cb), (defpot_vb, defpot_cb)


def get_defpot_corrections_ref(strain, defpots):
    defpots_vb_array = [defpots[0], defpots[0], 0.0]
    defpots_cb_array = [defpots[1], defpots[1], 0.0]
    strain_asarray = [strain[0, 0], strain[1, 1], 0.0]
    vb_correction = np.dot(defpots_vb_array, strain_asarray)
    cb_correction = np.dot(defpots_cb_array, strain_asarray)
    return vb_correction, cb_correction


def get_defpot_corrections(strain, defpots, soc: bool=False):
    from asr.moire.defpots import Result

    dpresults = Result.fromdict(read_json(defpots))
    if not soc:
        defpots_kb = np.asarray(dpresults.deformation_potentials_nosoc)
        edges_kb = np.asarray(dpresults.edges_nosoc)
    else:
        defpots_kb = np.asarray(dpresults.deformation_potentials_soc)
        edges_kb = np.asarray(dpresults.edges_soc)

    # Let's reshape the strain tensor in a way that is compatible with the deformation potentials
    avg_strain = (strain + strain.T) / 2
    strain_flat = [avg_strain[0, 0],
                   avg_strain[1, 1],
                   avg_strain[2, 2],
                   avg_strain[1, 2],
                   avg_strain[0, 2],
                   avg_strain[0, 1]]

    corrections_kb = np.zeros((edges_kb.shape[0], 2))
    for k, defpot_kb in enumerate(defpots_kb):
        for b, band in enumerate(['vb', 'cb']):
            defpot_b = defpot_kb[:, b]
            corrections_kb[k, b] = np.dot(defpot_b, strain_flat)

    newedges = edges_kb + corrections_kb
    whereis_new_vbm = np.argmax(newedges[:, 0])
    whereis_new_cbm = np.argmin(newedges[:, 1])

    return corrections_kb[whereis_new_vbm, 0], corrections_kb[whereis_new_cbm, 1]
        

    #defpots_vb_array = [defpots[0], defpots[0], 0.0]
    #defpots_cb_array = [defpots[1], defpots[1], 0.0]
    #strain_asarray = [strain[0, 0], strain[1, 1], 0.0]
    #vb_correction = np.dot(defpots_vb_array, strain_asarray)
    #cb_correction = np.dot(defpots_cb_array, strain_asarray)
    #return vb_correction, cb_correction


@command('asr.scs_shifts')
@option ('--info-file')
@option ('--output-file')
@option ('--defpots-a')
@option ('--defpots-b')
@option ('--soc', is_flag=True)
@option ('--add', is_flag=True)
@option ('--only-strain', is_flag=True)
@option ('--layer-info-file')
def main(info_file: str = 'makemoire-info.json',
         output_file: str='shifts.json', 
         defpots_a = '/home/niflheim2/steame/moire/testing/deformation-potentials/spec-kpts-only/MoSe2/0.2/results-asr.moire.defpots.json',
         defpots_b = '/home/niflheim2/steame/moire/testing/deformation-potentials/spec-kpts-only/WS2/0.2/results-asr.moire.defpots.json',
         layer_info_file: str='/home/niflheim2/steame/moire/utils/layer_info.json',
         soc: bool=False,
         only_strain: bool=False,
         add: bool=False):

    info = read_json(info_file)
    for key in info.keys():
        info.update({key: np.asarray(info[key])})
    shifts_a, _ = get_shifts_and_defpots(info['uid_a'], layer_info_file)
    shifts_b, _ = get_shifts_and_defpots(info['uid_b'], layer_info_file)

    if defpots_a and defpots_b:
        defpot_corr_a_vb, defpot_corr_a_cb = get_defpot_corrections(info['strain_a'], defpots_a, soc=soc)
        defpot_corr_b_vb, defpot_corr_b_cb = get_defpot_corrections(info['strain_b'], defpots_b, soc=soc)
        if add:
            shifts = {
                'shift_v1': shifts_a[0] + defpot_corr_a_vb,
                'shift_c1': shifts_a[1] + defpot_corr_a_cb,
                'shift_v2': shifts_b[0] + defpot_corr_b_vb,
                'shift_c2': shifts_b[1] + defpot_corr_b_cb
            }
        else:
            shifts = {
                'shift_v1': shifts_a[0] - defpot_corr_a_vb,
                'shift_c1': shifts_a[1] - defpot_corr_a_cb,
                'shift_v2': shifts_b[0] - defpot_corr_b_vb,
                'shift_c2': shifts_b[1] - defpot_corr_b_cb
            }
        write_json(output_file, shifts)
                
    elif not defpots_a and not defpots_b:
        shifts = {
            'shift_v1': shifts_a[0],
            'shift_c1': shifts_a[1],
            'shift_v2': shifts_b[0],
            'shift_c2': shifts_b[1]
        }
        write_json(output_file, shifts)

    else:
        raise ValueError('Either provide deformation potential result files for both layers or provide none.')


if __name__ == '__main__':
    main.cli()
