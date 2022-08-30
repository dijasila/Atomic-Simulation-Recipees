"""
This is a dummy recipe which stores a "label" (text string).
It's meant to be a specification of where the material came from.
For example, the "lattice decoration" materials in C2DB are labelled
as coming from "lattice decoration".
"""

from asr.core import command, option, ASRResult, DictStr


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, describe_entry

    label = result.get('label')
    if label is None:
        return []  # No panels generated

    tabulated_things = ['label']

    labeltable = table(
        row, 'Miscellaneous', tabulated_things, key_descriptions, 2)

    print('THE LABEl', label)

    panel = {'title': 'Summary',
             'columns': [[
                 labeltable,
                 {
                     'type': 'table',
                     'rows': [],  # no rows, what sense does this make?
                  # 'columnwidth': 4,
                  }]],
             }
    return [panel]


label_explanation = (
    'Label specifying generation procedure or origin of material')


class LabelResult(ASRResult):
    label: str
    key_descriptions = {'label': label_explanation}
    formats = {'ase_webpanel': webpanel}


@command(module='asr.utils.labels',
         returns=LabelResult)
@option('--label', help=label_explanation, type=str)
def main(label):
    return LabelResult.fromdata(label=label)


if __name__ == '__main__':
    main.cli()
