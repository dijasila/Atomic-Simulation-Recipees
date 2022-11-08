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


def arxiv(identifier):
    from asr.database.browser import href
    # (Avoid "browser"-imports at module level)
    return href(f'arXiv:{identifier}',
                f'https://arxiv.org/abs/{identifier}')


def doi(identifier):
    from asr.database.browser import href
    return href('doi:{identifier}', 'https://doi.org/{identifier}')


def webpanel(result, row, key_descriptions):
    from asr.database.browser import describe_entry

    lyngby22_link = arxiv('2206.12159')
    lyngby22_description_CDVAE = """\
DFT relaxed structures generated by the generative machine learning model:
Crystal Diffusion Variational AutoEncoder (CDVAE).

Ref: {link}
""".format(link=lyngby22_link)

    lyngby22_description_LDP = """\
DFT relaxed structures generated by elemental substitution.

Ref: {link}
""".format(link=lyngby22_link)

    lyngby22_description_training = """\
Training/seed structures for CDVAE and elemental substituion.

Ref: {link}
""".format(link=lyngby22_link)

    # Apply to whole push-manti-tree
    pushed02_22_description = """\
Materials were obtained by pushing dynamically unstable structures along
an unstable phonon mode followed by relaxation.

Ref: {link}
""".format(link=arxiv('2201.08091'))

    # Apply to whole ICSD-COD tree
    exfoliated02_21_description = """\
The materials were obtained by exfoliation of experimentally known layered
bulk crystals from the COD and ICSD databases.

Ref: {link}
""".format(link=doi('10.1088/2053-1583/ac1059'))

    decoration06_22_description = """\
Materials were obtained by systematic lattice decoration of thermodynamically
stable monolayers followed by relaxation."""
    # To be put on archive.  (It's probably there now.)

    # Apply to all materials with "class=Janus"
    janus10_19_description = """\
The materials generated by systematic lattice decoration and relaxation
using the MoSSe and BiTeI monolayers as seed structures.

Ref: {link}
""".format(link=doi('10.1021/acsnano.9b06698'))

    # All remaining materials, original c2db
    original03_18_description = """\
The materials constituted the first version of the C2DB.
They were obtained by lattice decoration of prototype monolayer crystals
known from experiments or earlier computational studies.

Ref: {link}
""".format(link=doi('10.1088/2053-1583/aacfc1'))

    descriptions = {
        'decoration06-22': decoration06_22_description,
        'exfoliated02-21': exfoliated02_21_description,
        'janus10-19': janus10_19_description,
        'Lyngby22_CDVAE': lyngby22_description_CDVAE,
        'Lyngby22_LDP': lyngby22_description_LDP,
        'Lyngby22_training': lyngby22_description_training,
        'original03-18': original03_18_description,
        'pushed02-22': pushed02_22_description,
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
def main(label: str) -> LabelResult:
    return LabelResult.fromdata(label=label)


if __name__ == '__main__':
    main.cli()
