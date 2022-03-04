from asr.core import command, ASRResult, prepare_result, chdir
import typing


def webpanel(result, row, key_description):
    from asr.database.browser import (WebPanel,
                                      table)

    baselink = 'http://gauss:5000/database.db/row/'
    charged_table = table(row, 'Charged systems', [])
    for element in result.chargedlinks:
        charged_table['rows'].extend(
            [[f'{element[1]}',
              f'<a href="{baselink}{element[0]}">link</a>']])

    neutral_table = table(row, 'Within the same material', [])
    for element in result.neutrallinks:
        neutral_table['rows'].extend(
            [[f'{element[1]}',
              f'<a href="{baselink}{element[0]}">link</a>']])
    for element in result.pristinelinks:
        neutral_table['rows'].extend(
            [[f'{element[1]}',
              f'<a href="{baselink}{element[0]}">link</a>']])

    panel = WebPanel('Related materials',
                     columns=[[charged_table], [neutral_table]],
                     sort=45)

    return [panel]


@prepare_result
class Result(ASRResult):
    """Container for defectlinks results."""
    chargedlinks: typing.List
    neutrallinks: typing.List
    pristinelinks: typing.List

    key_descriptions = dict(
        chargedlinks='Links tuple for the charged states of the same defect.',
        neutrallinks='Links tuple for other defects within the same material.',
        pristinelinks='Link for pristine material.')

    formats = {'ase_webpanel': webpanel}


@command(module='asr.defectlinks',
         requires=['structure.json'],
         dependencies=['asr.relax'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    """Create database links for the defect project."""
    from ase.formula import Formula
    from pathlib import Path
    from asr.core import read_json
    from asr.database.material_fingerprint import main as material_fingerprint
    p = Path('.')

    # First, get charged links for the same defect system
    chargedlinks = []
    chargedlist = list(p.glob('./../charge_*'))
    for charged in chargedlist:
        if (Path(charged / 'structure.json').is_file() and not
           str(charged.absolute()).endswith('charge_0')):
            with chdir(charged):
                res = material_fingerprint()
                # res = read_json('results-asr.database.material_fingerprint.json')
                uid = res['uid']
            host = Formula(str(charged.absolute()).split('/')[-4].split(
                'defects.')[-1].split('.')[0].split('_')[0])
            defect = split(str(charged.absolute()).split('/')[-4].split(
                '.')[-1].split('_'))
            charge = str(charged.absolute()).split('charge_')[-1]
            if defect[0] == 'v':
                defect[0] = 'V'
            defect = f"{defect[0]}<sub>{defect[1]}</sub> (charge {charge})"
            host = f"{host:html}"
            chargedlinks.append((uid, f"{defect} in {host}"))

    # Second, get the neutral links of systems in the same host
    neutrallinks = []
    neutrallist = list(p.glob('./../../*/charge_0'))
    for neutral in neutrallist:
        if (Path(neutral / 'structure.json').is_file()):
            with chdir(neutral):
                res = material_fingerprint()
                # res = read_json('results-asr.database.material_fingerprint.json')
                uid = res['uid']
            host = Formula(str(neutral.absolute()).split('/')[-2].split(
                'defects.')[-1].split('.')[0].split('_')[0])
            defect = split(str(neutral.absolute()).split('/')[-2].split(
                '.')[-1].split('_'))
            charge = str(neutral.absolute()).split('charge_')[-1]
            if defect[0] == 'v':
                defect[0] = 'V'
            defect = f"{defect[0]}<sub>{defect[1]}</sub> (charge {charge})"
            host = f"{host:html}"
            neutrallinks.append((uid, f"{defect} in {host}"))

    # Third, the pristine material
    pristinelinks = []
    pristine = list(p.glob('./../../defects.pristine_sc*'))[0]
    if (Path(pristine / 'structure.json').is_file()):
        with chdir(pristine):
            res = material_fingerprint()
            # res = read_json('results-asr.database.material_fingerprint.json')
            uid = res['uid']
        pristinelinks.append((uid, f"pristine material"))

    return Result.fromdata(
        chargedlinks=chargedlinks,
        neutrallinks=neutrallinks,
        pristinelinks=pristinelinks)


def split(word):
    return [char for char in word]


if __name__ == '__main__':
    main.cli()
