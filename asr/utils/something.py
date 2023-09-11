"""Template recipe."""
import json
from pathlib import Path
from asr.core import command, option, ASRResult


@command('asr.something')
@option('--number', type=int)
def main(number=5) -> ASRResult:
    """Calculate something."""
    something = calculate_something(number)
    results = {'number': number,
               'something': something}
    Path('something.json').write_text(json.dumps(results))
    return results


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


def webpanel(result, row, key_descriptions):
    from asr.webpages.browser import fig, create_table

    if 'something' not in row.data:
        return None, []

    table1 = create_table(
        row=row, header=['Property', 'Value'], keys=['something'],
        key_descriptions=key_descriptions, digits=2)
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


group = 'property'
creates = ['something.json']  # what files are created
dependencies = []  # no dependencies
resources = '1:10m'  # 1 core for 10 minutes
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()
