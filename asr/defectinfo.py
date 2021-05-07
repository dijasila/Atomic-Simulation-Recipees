from asr.core import command, ASRResult, prepare_result, read_json
import typing
from pathlib import Path


@prepare_result
class Result(ASRResult):
    """Container for asr.defectinfo results."""
    defect_name: str
    host_name: str
    charge_state: str
    host_pointgroup: str
    host_spacegroup: str
    host_crystal: str
    host_uid: str


    key_descriptions: typing.Dict[str, str] = dict(
        defect_name='Name of the defect({type}_{position}).',
        host_name='Name of the host system.',
        charge_state='Charge state of the defect system.',
        host_pointgroup='Point group of the host crystal.',
        host_spacegroup='Space group of the host crystal.',
        host_crystal='Crystal type of the host crystal.',
        host_uid='UID of the primitive host crystal.')


@command(module='asr.defectinfo',
         resources='1:10m',
         returns=Result)
def main() -> Result:
    """Extract defect and host name.

    Extract defect name and host hame based on the folder structure
    created by asr.setup.defects."""
    p = Path('.')
    pathstr = str(p.absolute())

    if pathstr.split('/')[-1].startswith('defects.pristine_sc'):
        host_name = pathstr.split('/')[-2].split('-')[0]
        defect_name = 'pristine'
        charge_state = ''
        prispath = p
        primpath = Path(p / '../')
    elif pathstr.split('/')[-1].startswith('charge'):
        host_name = pathstr.split('/')[-3].split('-')[0]
        defect_name = pathstr.split('/')[-2].split('.')[-1]
        charge_state = f"(charge {pathstr.split('/')[-1].split('_')[-1]})"
        prispath = list(p.glob('../../defects.pristine_sc*'))[-1]
        primpath = Path(p / '../../')
    else:
        raise ValueError('ERROR: needs asr.setup.defects to extract'
                         ' information on the defect system. Furthermore, '
                         'asr.structureinfo needs to run for the pristine '
                         'system and asr.database.material_fingerprint for '
                         'the primitive structure.')

    resfile = Path(prispath / 'results-asr.structureinfo.json')
    if resfile.is_file():
        res = read_json(resfile)
        host_pointgroup = res['pointgroup']
        host_spacegroup = res['spacegroup']
        host_crystal = res['crystal_type']
    else:
        print('WARNING: no asr.structureinfo ran for the pristine system!')
        host_pointgroup = None
        host_spacegroup = None
        host_crystal = None

    primres = Path(primpath / 'results-asr.database.material_fingerprint.json')
    if primres.is_file():
        res = read_json(primres)
        host_uid = res['uid']
    else:
        print('WARNING: no asr.database.material_fingerprint ran for primitive structure!')
        host_uis = None



    return Result.fromdata(
        defect_name=defect_name,
        host_name=host_name,
        charge_state=charge_state,
        host_pointgroup=host_pointgroup,
        host_spacegroup=host_spacegroup,
        host_crystal=host_crystal,
        host_uid=host_uid)


if __name__ == '__main__':
    main.cli()
