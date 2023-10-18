from pathlib import Path
from ase.db import connect
from ase.io import read
from asr.core import command, read_json, option
from asr.defectlinks import get_charge_from_folder
from asr.defect_symmetry import DefectInfo
from asr.setup.defects import return_distances_cell
from asr.paneldata import DefectInfoResult


@command(module='asr.defectinfo',
         resources='1:10m',
         returns=DefectInfoResult)
@option('--structurefile', help='Structure file for the evaluation of '
        'the nearest neighbor distance.', type=str)
@option('--pristine/--no-pristine', help='Flag to treat systems '
        'in the pristine folder of the set up tree structure '
        'from asr.setup.defects.', is_flag=True)
@option('--dbpath', help='Path to C2DB database file for host '
        'property extraction.', type=str)
def main(structurefile: str = 'structure.json',
         pristine: bool = False,
         dbpath: str = '/home/niflheim/fafb/db/c2db_july20.db') -> DefectInfoResult:
    """Extract defect, host name, and host crystalproperties.

    Extract defect name and host hame based on the folder structure
    created by asr.setup.defects.
    """
    atoms = read(structurefile)
    # extract path of the current directory
    p = Path('.')

    # collect all relevant paths for defect info extraction
    primitivepath, pristinepath = get_primitive_pristine_folderpaths(
        p, pristine)

    # obtain defectname and chargestate string
    if pristine:
        defectname = 'pristine'
        chargestate = ''
    else:
        defectname, chargestate = get_defectinfo_from_path(p)

    # calculation nearest neighbor distance for supercell structure
    R_nn = get_nearest_neighbor_distance(atoms)

    # extract atomic structure and name of the host crystal
    hostatoms = read(primitivepath / 'unrelaxed.json')
    hostname = hostatoms.get_chemical_formula()

    # collect all relevant results files
    resultsfile = Path(pristinepath / 'results-asr.structureinfo.json')
    if resultsfile.is_file():
        res = read_json(resultsfile)
        host_pointgroup = res['pointgroup']
        host_spacegroup = res['spacegroup']
        host_crystal = res['crystal_type']
    else:
        raise FileNotFoundError(
            f'did not find {resultsfile.name} in {pristinepath}!')

    resultsfile = Path(primitivepath / 'results-asr.database.material_fingerprint.json')
    if resultsfile.is_file():
        res = read_json(resultsfile)
        host_uid = res['uid']
    else:
        raise FileNotFoundError(
            f'Did not find {resultsfile.name} in {primitivepath}!')

    # extract host crystal properties from C2DB
    db = connect(dbpath)
    hof, pbe, hse = get_host_properties_from_db(db, host_uid)

    return DefectInfoResult.fromdata(
        defect_name=defectname,
        host_name=hostname,
        charge_state=chargestate,
        host_pointgroup=host_pointgroup,
        host_spacegroup=host_spacegroup,
        host_crystal=host_crystal,
        host_uid=host_uid,
        host_hof=hof,
        host_gap_pbe=pbe,
        host_gap_hse=hse,
        R_nn=R_nn)


def get_defectinfo_from_path(path):
    defectinfo = DefectInfo(defectpath=path)
    defectname = defectinfo.defecttoken
    charge = get_charge_from_folder(path)
    chargestate = f'(charge {charge})'

    return defectname, chargestate


def get_primitive_pristine_folderpaths(path, pristine):
    if pristine:
        primitivepath = Path('../')
        pristinepath = path
    else:
        primitivepath = Path('../../')
        pristinepath = list(path.glob('../../defects.pristine_sc*'))[-1]

    return primitivepath, pristinepath


def get_nearest_neighbor_distance(atoms):
    cell = atoms.get_cell()
    distances = return_distances_cell(cell)

    return min(distances)


def get_host_properties_from_db(db, uid):
    """Extract host properties from C2DB.

    The input database needs a uid, hform, and gap keyword.
    """
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
