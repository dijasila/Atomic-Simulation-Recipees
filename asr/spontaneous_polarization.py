"""Topological analysis of electronic structure."""
from asr.core import command, ASRResult, prepare_result
from asr.database.browser import describe_entry, WebPanel, href

kruse_title = ('M. Kruse et al. Two-dimensional ferroelectrics from '
               'high throughput computational screening '
               'npj Computational Materials 9, 45 (2023)')
kruse_doi = 'https://doi.org/10.1038/s41524-023-00999-5'


def webpanel(result, row, key_descriptions):
    from asr.utils.hacks import gs_xcname_from_row

    xcname = gs_xcname_from_row(row)
    description = ('An insulating path connecting the polar ground state to a '
                   'non-polar structure has been identified. This allows for a '
                   'unique determination of the spontaneous polarization, '
                   'which is stated under basic electronic properties '
                   'below.\n\n' + href(kruse_title, kruse_doi))
    datarow = [describe_entry('Ferroelectric', description), result.Ferroelectric]

    summary = WebPanel(title='Summary',
                       columns=[[{'type': 'table',
                                  'header': ['Basic properties', ''],
                                  'rows': [datarow]}]])

    descriptionx = ('x-component of spontaneous polarization')
    datarow_Px = [describe_entry('Px', descriptionx), f'{result.Px:0.2f} pC/m']

    descriptiony = ('y-component of spontaneous polarization')
    datarow_Py = [describe_entry('Py', descriptiony), f'{result.Py:0.2f} pC/m']

    descriptionz = ('z-component of spontaneous polarization')
    datarow_Pz = [describe_entry('Pz', descriptionz), f'{result.Pz:0.2f} pC/m']

    basicelec = WebPanel(title=f'Basic electronic properties ({xcname})',
                         columns=[[{'type': 'table',
                                    'header': ['Property', ''],
                                    'rows': [datarow_Px,
                                             datarow_Py,
                                             datarow_Pz]}]],
                         sort=15)

    return [summary, basicelec]


@prepare_result
class Result(ASRResult):

    Ferroelectric: bool
    Px: float
    Py: float
    Pz: float
    P: float

    key_descriptions = {
        'Ferroelectric': ('Material has switchable polarization'),
        'Px': ('x-component of spontaneous polarization [pC / m]'),
        'Py': ('y-component of spontaneous polarization [pC / m]'),
        'Pz': ('z-component of spontaneous polarization [pC / m]'),
        'P': ('Magnitude of spontaneous polarization [pC / m]'),
    }

    formats = {"ase_webpanel": webpanel}


@command(module='asr.spontaneous_polarization',
         returns=Result)
def main() -> Result:
    data = {}

    f = open('polarization.dat', 'r')
    Px = eval(f.readline())
    Py = eval(f.readline())
    Pz = eval(f.readline())
    f.close()

    data['Ferroelectric'] = True
    data['Px'] = Px
    data['Py'] = Py
    data['Pz'] = Pz
    data['P'] = (Px**2 + Py**2 + Pz**2)**0.5

    return data


if __name__ == '__main__':
    main.cli()
