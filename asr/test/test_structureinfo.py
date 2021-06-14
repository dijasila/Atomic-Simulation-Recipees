import typing
from asr.core import ASRResult
from asr.structureinfo import Result

skip_keys = {'version', 'prev_version', 'key_descriptions'}


def fill_in_arbitrary_result_data(cls):

    type_hints = typing.get_type_hints(cls)

    data = {}
    for key, tp in type_hints.items():
        if key in skip_keys:
            continue
        if issubclass(tp, ASRResult):
            data[key] = fill_in_arbitrary_result_data(tp)
        elif issubclass(tp, typing.Tuple):
            data[key] = tuple(tp2(2) for tp2 in tp.__args__)
        elif tp == dict:
            data[key] = {'key': 'value'}
        else:
            data[key] = tp(2)

    return cls(data=data)


class Row:

    def __init__(self, result, data):
        self.data = data
        self.result = result
        self.cod_id = 'some_cod_id'
        self.icsd_id = 'some_icsd_id'
        self.doi = 'some_doi'
        setattr(self, 'class', 'some class')

    def get(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return getattr(self.result, key)


def test_cod_id():
    result = fill_in_arbitrary_result_data(Result)

    row = Row(
        result=result,
        data={'results-asr.structureinfo.json': result},
    )
    # XXX We are trying to decouple from Row objects
    webpanels = result.format_as('ase_webpanel', vars(row), {})
    webpanel = webpanels[0]
    tablerows = webpanel['columns'][0][0]['rows']
    keys = [row[0] for row in tablerows]
    assert 'cod_id' in keys
