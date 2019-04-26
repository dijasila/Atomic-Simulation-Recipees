"""Template recipe."""
import json
from pathlib import Path
from asr.utils import option, update_defaults
import click


@click.command()
@update_defaults('asr.template_recipe')
@option('--number', default=5)
def main(number):
    """Calculate something."""
    something = calculate_something(number)
    results = {'number': number,
               'somehting': something}
    Path('something.json').write_text(json.dumps(results))


def calculate_something(number):
    return number + 2


def collect_data(atoms):
    path = Path('something.json')
    if not path.is_file():
        return {}, {}, {}
    # Read data:
    dct = json.loads(path.read_text())
    # Define key-value pairs, key descriptions and data:
    kvp = {'something': dct['something']}
    kd = {'something': ('Something', 'Longer description', 'unit')}
    data = {'something':
            {'stuff': 'more complicated data structures',
             'things': [0, 1, 2, 1, 0]}}
    return kvp, kd, data


def webpanel(row, key_descriptions):
    from asr.custom import fig, table

    if 'something' not in row.data:
        return None, []

    table1 = table(row,
                   'Property',
                   ['something'],
                   kd=key_descriptions)
    panel = ('Title',
             [[fig('something.png'), table1]])
    things = [(create_plot, ['something.png'])]
    return panel, things


def create_plot(row, fname):
    import matplotlib.pyplot as plt

    data = row.data.something
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data.things)
    plt.savefig(fname)


group = 'Property'
creates = ['something.json']
dependencies = []

if __name__ == '__main__':
    main(standalone_mode=False)
