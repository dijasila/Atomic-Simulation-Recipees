from asr.core import command, ASRResult, prepare_result
import typing
from pathlib import Path


@prepare_result
class Result(ASRResult):
    """Container for asr.defectinfo results."""
    defect_name: str
    host_name: str

    key_descriptions: typing.Dict[str, str] = dict(
        defect_name='Name of the defect({type}_{position}).',
        host_name='Name of the host system.')


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
    elif pathstr.split('/')[-1].startswith('charge'):
        host_name = pathstr.split('/')[-3].split('-')[0]
        defect_name = pathstr.split('/')[-2].split('.')[-1]
    else:
        raise ValueError('ERROR: needs asr.setup.defects to extract'
                         ' information on the defect system.')

    return Result.fromdata(
        defect_name=defect_name,
        host_name=host_name)


if __name__ == '__main__':
    main.cli()
