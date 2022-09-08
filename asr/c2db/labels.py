"""
This is a dummy recipe which stores a "label" (text string).
It's meant to be a specification of where the material came from.
For example, the "lattice decoration" materials in C2DB are labelled
as coming from "lattice decoration".

The web panel hardcodes different labels, associating them with
specific descriptions, links and so on.

So the web panel is not general.
"""

from asr.core import command, option, ASRResult


def webpanel(result, row, key_descriptions):
    from asr.database.browser import describe_entry, href
    lyngby22_description_CDVAE = """\
    DFT relaxed structures generated by the generative machine learning model: Crystal Diffusion Variational AutoEncoder (CDVAE). 

    Ref.: href('paper', 'https://arxiv.org/abs/2206.12159')
    """
    lyngby22_description_LDP = """\
    DFT relaxed structures generated by elemental substitution. 

    Ref.: href('paper', 'https://arxiv.org/abs/2206.12159')
    """
    # (The "potato" description serves as an example of how to define
    #  information in web panels)
    potato_link = href('potato', 'https://en.wikipedia.org/wiki/Potato')
    descriptions = {
        'potato': (f'The {potato_link} is a starchy tuber of the plant '
                   'Solanum tuberosum'),
        'Lyngby22_CDVAE': lyngby22_description_CDVAE,
        'Lyngby22_LDP': lyngby22_description_LDP,
    }

    label = result.get('label')
    if label in descriptions:
        label = describe_entry(label, descriptions[label])
    if label is None:
        return []  # No panels generated

    entryname = describe_entry('Origin', label_explanation)

    panel = {
        'title': 'Summary',
        'columns': [[{
            'type': 'table',
            'rows': [[entryname, label]],
            'columnwidth': 4,
        }]],
    }
    return [panel]


label_explanation = (
    'Label specifying generation procedure or origin of material')


class LabelResult(ASRResult):
    label: str
    key_descriptions = {'label': label_explanation}
    formats = {'ase_webpanel': webpanel}


@command(module='asr.c2db.labels',
         returns=LabelResult)
@option('--label', help=label_explanation, type=str)
def main(label):
    return LabelResult.fromdata(label=label)


if __name__ == '__main__':
    main.cli()
