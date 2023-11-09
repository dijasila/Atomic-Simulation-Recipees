from pathlib import Path
from ase.formula import Formula
from asr.core import (command, chdir, read_json)
from asr.database.material_fingerprint import main as material_fingerprint
from asr.defect_symmetry import DefectInfo
from asr.result.resultdata import DefectLinksResult


@command(module='asr.defectlinks',
         # requires=['structure.json'],
         # dependencies=['asr.relax'],
         resources='1:1h',
         returns=DefectLinksResult)
def main() -> DefectLinksResult:
    """Generate QPOD database links for the defect project.

    This recipe is dependent on the folder structure created by
    asr.setup.defects. Will generate links between different chargestates
    of the same defect, neutral chargestates for different defects of the
    same host material, and pristine structures. For your own project,
    make sure to change the 'basepath' variable in the webpanel function
    according to the location of your database.
    """
    # extract path of current directory
    p = Path('.')

    # First, get charged links for the same defect system
    chargedlinks = []
    chargedlist = list(p.glob('./../charge_*'))
    for charged in chargedlist:
        chargedlinks = get_list_of_links(charged)

    # Second, get the neutral links of systems in the same host
    neutrallinks = []
    neutrallist = list(p.glob('./../../*/charge_0'))
    for neutral in neutrallist:
        neutrallinks = get_list_of_links(neutral)

    # Third, the pristine material
    pristinelinks = []
    pristine = list(p.glob('./../../defects.pristine_sc*'))[0]
    if (Path(pristine / 'structure.json').is_file()):
        uid = get_uid_from_fingerprint(pristine)
        pristinelinks.append((uid, 'pristine material'))

    return DefectLinksResult.fromdata(
        chargedlinks=chargedlinks,
        neutrallinks=neutrallinks,
        pristinelinks=pristinelinks)


def get_list_of_links(path):
    links = []
    structurefile = path / 'structure.json'
    charge = get_charge_from_folder(path)
    if structurefile.is_file() and charge != 0:
        defectinfo = DefectInfo(defectpath=path)
        uid = get_uid_from_fingerprint(path)
        hostformula = get_hostformula_from_defectpath(path)
        defectstring = get_defectstring_from_defectinfo(defectinfo, charge)
        links.append((uid, f"{defectstring} in {hostformula:html}"))

    return links


def get_uid_from_fingerprint(path):
    with chdir(path):
        material_fingerprint()
        res = read_json('results-asr.database.material_fingerprint.json')
        uid = res['uid']

    return uid


def get_defectstring_from_defectinfo(defectinfo, charge):
    defectstring = ''
    for name in defectinfo.names:
        def_type, def_kind = defectinfo.get_defect_type_and_kind_from_defectname(
            name)
        defectstring += f"{def_type}<sub>{def_kind}</sub>"
    defectstring += f" (charge {charge})"

    return defectstring


def get_hostformula_from_defectpath(path):
    fullpath = path.absolute()
    token = fullpath.parent.name
    hostname = token.split('_')[0].split('defects.')[-1]

    return Formula(hostname)


def get_charge_from_folder(path):
    fullpath = path.absolute()
    chargedstring = fullpath.name
    charge = int(chargedstring.split('charge_')[-1])

    return charge


if __name__ == '__main__':
    main.cli()
