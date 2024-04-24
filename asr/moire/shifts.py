import numpy as np
from ase.io import read
from ase.io.jsonio import read_json, write_json
from ase.db import connect
from asr.core import command, option


def get_shifts_and_defpots(uid, c2db):
    #querry = "uid=\""+str(uid)+"\""
    querry = 'uid='+str(uid)
    print(querry)
    mrow = c2db.get(querry)

    GW_VBM = mrow.data["results-asr.gw.json"]["kwargs"]["data"]["vbm_gw_nosoc"]
    GW_CBM = mrow.data["results-asr.gw.json"]["kwargs"]["data"]["cbm_gw_nosoc"]

    DFT_bs = mrow.data["results-asr.bandstructure.json"]['kwargs']['data']['bs_nosoc']
    DFT_ef = DFT_bs['efermi']
    DFT_e = DFT_bs['energies'][0]
    vn = np.sum(np.all(DFT_e < DFT_ef, axis=0))
    DFT_VBM = np.max(DFT_e[:, vn-1])
    DFT_CBM = np.min(DFT_e[:, vn])

    shift_vb = GW_VBM - DFT_VBM
    shift_cb = GW_CBM - DFT_CBM

    defpots = mrow.data["results-asr.deformationpotentials.json"]['kwargs']['data']
    return (shift_vb, shift_cb), defpots


def get_strain_corrections(strain, defpots, subtract: bool=True, soc: bool=False):

    if soc:
        defpots = defpots['defpots_soc']
    else:
        defpots = defpots['defpots_nosoc']

    # Let's reshape the strain tensor in a way that is compatible with the deformation potentials
    avg_strain = (strain + strain.T) / 2
    strain_flat = [avg_strain[i] for i in [(0, 0), (1, 1), (0, 1)]]

    corr_vbm = 0
    corr_cbm = 0

    for c, coord in enumerate(['xx', 'yy', 'xy']):
        corr_vbm += defpots['VBM'][coord]['VB'] * strain_flat[c]
        corr_cbm += defpots['CBM'][coord]['CB'] * strain_flat[c]

    if not subtract:
        return corr_vbm, corr_cbm
    else:
        return -corr_vbm, -corr_cbm
        

def get_shifts(info, database, correct_strain, subtract, soc, only_strain):
    c2db = connect(database)
    shifts_a, defpots_a = get_shifts_and_defpots(info['uid_a'], c2db)
    shifts_b, defpots_b = get_shifts_and_defpots(info['uid_b'], c2db)
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
@option ('--database')
@option ('-o', '--output-file')
@option ('--soc', is_flag=True)
@option ('--correct-strain/--dont-correct-strain', is_flag=True)
@option ('--subtract/--add', is_flag=True)
@option ('--only-strain', is_flag=True)
def main(info_file: str = 'bilayer-info.json',
         output_file: str='shifts.json', 
         correct_strain: bool=True,
         database: str='/home/niflheim2/cmr/databases/c2db/c2db-first-class-20240102.db',
         soc: bool=False,
         only_strain: bool=False,
         subtract: bool=True):

    info = read_json(info_file)
    for key in info.keys():
        info.update({key: np.asarray(info[key])})

    shifts = get_shifts(info, database, correct_strain, subtract, soc, only_strain)
    shifts_dct = {}
    for i, key in enumerate(['shift_v1', 'shift_c1', 'shift_v2', 'shift_c2']):
        shifts_dct.update({key: shifts[i]})

    write_json(output_file, shifts_dct)


if __name__ == '__main__':
    main.cli()
