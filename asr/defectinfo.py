from asr.core import command, ASRResult, prepare_result, read_json
import typing
from pathlib import Path


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (table, describe_entry, code, bold,
                                      br, href, dl, div)

    spglib = href('SpgLib', 'https://spglib.github.io/spglib/')
    crystal_type = describe_entry(
        'Host crystal type',
        "The crystal type is defined as "
        + br
        + div(bold('-'.join([code('stoi'),
                             code('spg no.'),
                             code('occ. wyck. pos.')])), 'well well-sm text-center')
        + 'where'
        + dl(
            [
                [code('stoi'), 'Stoichiometry.'],
                [code('spg no.'), f'The spacegroup calculated with {spglib}.'],
                [code('occ. wyck. pos.'),
                 'Alphabetically sorted list of occupied '
                 f'wyckoff positions determined with {spglib}.'],
            ]
        )
    )

    spg_list_link = href(
        'space group', 'https://en.wikipedia.org/wiki/List_of_space_groups')
    spacegroup = describe_entry(
        'Host space group',
        f"The {spg_list_link} is determined with {spglib}.")
    pointgroup = describe_entry(
        'Host point group',
        f"The point group is determined with {spglib}.")
    host_hof = describe_entry(
        'Host heat of formation',
        result.key_descriptions['host_hof'])
    host_gap_pbe = describe_entry(
        'Host PBE bandgap',
        result.key_descriptions['host_gap_pbe'])
    host_gap_hse = describe_entry(
        'Host HSE bandgap',
        result.key_descriptions['host_gap_hse'])
    R_nn = describe_entry(
        'Defect nearest neighbor distance',
        result.key_descriptions['R_nn'])

    uid = result.host_uid
    uidstring = describe_entry(
        'C2DB link',
        'Link to C2DB entry of the host material.')

    basictable = table(result, 'Pristine crystal', [])
    basictable['rows'].extend(
        [[crystal_type, result.host_crystal]])
    basictable['rows'].extend(
        [[spacegroup, result.host_spacegroup]])
    basictable['rows'].extend(
        [[pointgroup, result.host_pointgroup]])
    basictable['rows'].extend(
        [[host_hof, f'{result.host_hof:.2f} eV/atom']])
    basictable['rows'].extend(
        [[host_gap_pbe, f'{result.host_gap_pbe:.2f} eV']])
    if result.host_gap_hse is not None:
        basictable['rows'].extend(
            [[host_gap_hse, f'{result.host_gap_hse:.2f} eV']])
    if result.R_nn is not None:
        basictable['rows'].extend(
            [[R_nn, f'{result.R_nn:.2f} Å']])

    if uid:
        basictable['rows'].extend(
            [[uidstring,
              '<a href="https://cmrdb.fysik.dtu.dk/c2db/row/{uid}"'
              '>{uid}</a>'.format(uid=uid)]])

    panel = {'title': 'Summary',
             'columns': [[basictable], []],
             'sort': -1}

    return [panel]


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
    host_hof: float
    host_gap_pbe: float
    host_gap_hse: float
    R_nn: float

    key_descriptions: typing.Dict[str, str] = dict(
        defect_name='Name of the defect({type}_{position}).',
        host_name='Name of the host system.',
        charge_state='Charge state of the defect system.',
        host_pointgroup='Point group of the host crystal.',
        host_spacegroup='Space group of the host crystal.',
        host_crystal='Crystal type of the host crystal.',
        host_uid='UID of the primitive host crystal.',
        host_hof='Heat of formation for the host crystal [eV/atom].',
        host_gap_pbe='PBE bandgap of the host crystal [eV].',
        host_gap_hse='HSE bandgap of the host crystal [eV].',
        R_nn='Nearest neighbor distance of repeated defects [Å].')

    formats = {"ase_webpanel": webpanel}


@command(module='asr.defectinfo',
         resources='1:10m',
         returns=Result)
def main() -> Result:
    """Extract defect and host name.

    Extract defect name and host hame based on the folder structure
    created by asr.setup.defects."""
    p = Path('.')
    pathstr = str(p.absolute())

    R_nn = None
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
        R_nn = get_nearest_neighbor_distance()
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
        raise ValueError('Error: no asr.database.material_fingerprint ran for '
                         'primitive structure! Make sure to run it before '
                         'running asr.defectinfo.')

    # extract host crystal properties from C2DB
    hof, pbe, hse = get_host_properties_from_C2DB(host_uid)

    return Result.fromdata(
        defect_name=defect_name,
        host_name=host_name,
        charge_state=charge_state,
        host_pointgroup=host_pointgroup,
        host_spacegroup=host_spacegroup,
        host_crystal=host_crystal,
        host_uid=host_uid,
        host_hof=hof,
        host_gap_pbe=pbe,
        host_gap_hse=hse,
        R_nn=R_nn)


def get_nearest_neighbor_distance():
    from ase.io import read
    import numpy as np

    try:
        atoms = read('structure.json')
    except FileNotFoundError:
        atoms = read('unrelaxed.json')
    cell = atoms.get_cell()
    distance_xx = np.sqrt(cell[0][0]**2 + cell[0][1]**2 + cell[0][2]**2)
    distance_yy = np.sqrt(cell[1][0]**2 + cell[1][1]**2 + cell[1][2]**2)
    distance_xy = np.sqrt((
        cell[0][0] + cell[1][0])**2 + (
        cell[0][1] + cell[1][1])**2 + (
        cell[0][2] + cell[1][2])**2)
    distance_mxy = np.sqrt((
        -cell[0][0] + cell[1][0])**2 + (
        -cell[0][1] + cell[1][1])**2 + (
        -cell[0][2] + cell[1][2])**2)
    distances = [distance_xx, distance_yy, distance_xy, distance_mxy]

    return min(distances)


def get_host_properties_from_C2DB(uid):
    """Extract host properties from C2DB."""
    from ase.db import connect

    db = connect('/home/niflheim/fafb/db/c2db_july20.db')

    for row in db.select(uid=uid):
        hof = row.hform
        gap_pbe = row.gap
        try:
            gap_hse = row.gap_hse
        except AttributeError:
            gap_hse = None

    return hof, gap_pbe, gap_hse


if __name__ == '__main__':
    main.cli()
