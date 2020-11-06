from asr.core import ASRResult, prepare_result
from asr.database.webpanel import WebPanel
import typing


def webpanel(result, row, key_descriptions):
    from asr.database.browser import describe_entry, entry_parameter_description
    parameter_description = entry_parameter_description(row.data, 'asr.gs@calculate')

    energy = describe_entry(result.energy, parameter_description)
    prop_table = {'type': 'table',
                  'header': ['Property', 'value'],
                  'rows': [['energy', energy]]}

    return [
        WebPanel(title='Title of my webpanel',
                 columns=[[prop_table], []])
    ]


@prepare_result
class Result(ASRResult):
    """My ground state results object.

    These results are generated using...
    """

    energy: float
    forces: typing.List[typing.tuple[float, float, float]]

    key_descriptions = dict(
        energy='The energy of the material.',
        forces='The forces on the atoms of the material.',
    )

    formats = {'ase_webpanel': webpanel}
