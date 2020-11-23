"""ASR command line interface."""
import sys
import os
from typing import Union, Dict, Any, List
import asr
from asr.core import read_json, chdir, ASRCommand, DictStr, set_defaults
import click
from pathlib import Path
import subprocess
from ase.parallel import parprint
from functools import partial
import importlib
from contextlib import contextmanager
import pickle


prt = partial(parprint, flush=True)


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


# def format(content, indent=0, title=None, pad=2):
#     colwidth_c = []
#     for row in content:
#         if isinstance(row, str):
#             continue
#         for c, element in enumerate(row):
#             nchar = len(element)
#             try:
#                 colwidth_c[c] = max(colwidth_c[c], nchar)
#             except IndexError:
#                 colwidth_c.append(nchar)

#     output = ''
#     if title:
#         output = f'\n{title}\n'
#     for row in content:
#         out = ' ' * indent
#         if isinstance(row, str):
#             output += f'{row}'
#             continue
#         for colw, desc in zip(colwidth_c, row):
#             out += f'{desc: <{colw}}' + ' ' * pad
#         output += out
#         output += '\n'

#     return output


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=asr.__version__)
def cli():
    ...


@cli.command()
@click.argument('command', nargs=1)
@click.argument('folders', nargs=-1)
@click.option('-n', '--not-recipe', is_flag=True,
              help='COMMAND is not a recipe.')
@click.option('-z', '--dry-run', is_flag=True,
              help='Show what would happen without doing anything.')
@click.option('-j', '--njobs', type=int, default=1,
              help='Run COMMAND in serial on JOBS processes.')
@click.option('-S', '--skip-if-done', is_flag=True,
              help='Skip execution of recipe if done.')
@click.option('--dont-raise', is_flag=True, default=False,
              help='Continue to next folder when encountering error.')
@click.option('--update', is_flag=True, default=False,
              help="Update existing results files. "
              "Only runs a recipe if it is already done.")
@click.option('--must-exist', type=str,
              help="Skip folder where this file doesn't exist.")
@click.option('--defaults', type=DictStr(),
              help="Set default parameters. Takes precedence over params.json.")
@click.option('--pipe', is_flag=True,
              help="Write result to stdout.")
@click.pass_context
def run(ctx, command, folders, not_recipe, dry_run, njobs,
        skip_if_done, dont_raise, update, must_exist,
        defaults, pipe):
    r"""Run recipe or python function in multiple folders.

    Examples
    --------
    Run the relax recipe
    >>> asr run relax

    Run the calculate function in the gs module
    >>> asr run gs@calculate

    Get help for a recipe
    >>> asr run "relax -h"

    Specify an argument
    >>> asr run "relax --ecut 600"

    Run relax recipe in two folders sequentially
    >>> asr run relax folder1/ folder2/

    """
    import multiprocessing

    nfolders = len(folders)
    if not folders:
        folders = ['.']
    else:
        prt(f'Number of folders: {nfolders}')

    if update:
        assert not skip_if_done

    kwargs = {
        'update': update,
        'skip_if_done': skip_if_done,
        'dont_raise': dont_raise,
        'dry_run': dry_run,
        'not_recipe': not_recipe,
        'command': command,
        'must_exist': must_exist,
        'defaults': defaults,
        'pipe': pipe,
    }
    if njobs > 1:
        processes = []
        for job in range(njobs):
            kwargs['job_num'] = job
            proc = multiprocessing.Process(
                target=run_command,
                args=(folders[job::njobs], ),
                kwargs=kwargs,
            )
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
            assert proc.exitcode == 0
    else:
        run_command(folders, **kwargs)


def append_job(string: str, job_num: Union[int, None] = None):
    """Append job number to message if provided."""
    if job_num is None:
        return string
    else:
        return f'Job #{job_num}: {string}'


def run_command(folders, *, command: str, not_recipe: bool, dry_run: bool,
                skip_if_done: bool, dont_raise: bool,
                job_num: Union[int, None] = None,
                update: bool = False,
                must_exist: Union[str, None] = None,
                defaults: Dict[str, Any],
                pipe: bool = False):
    """Run command in folders."""
    nfolders = len(folders)
    module, *args = command.split()
    function = None
    if '@' in module:
        module, function = module.split('@')

    if update:
        assert not skip_if_done, \
            append_job('Cannot combine --update with --skip-if-done.',
                       job_num=job_num)

    if not_recipe:
        assert function, \
            append_job('If this is not a recipe you have to specify a '
                       'specific function to execute.', job_num=job_num)
    else:
        if not module.startswith('asr.'):
            module = f'asr.{module}'

    if not function:
        function = 'main'

    mod = importlib.import_module(module)
    assert hasattr(mod, function), \
        append_job(f'{module}@{function} doesn\'t exist.', job_num=job_num)
    func = getattr(mod, function)

    if isinstance(func, ASRCommand):
        is_asr_command = True
    else:
        is_asr_command = False

    if dry_run:
        prt(append_job(f'Would run {module}@{function} '
                       f'in {nfolders} folders.', job_num=job_num))
        return

    for i, folder in enumerate(folders):
        with chdir(Path(folder)):
            try:
                if skip_if_done and func.done:
                    continue
                elif update and not func.done:
                    continue
                elif must_exist and not Path(must_exist).exists():
                    continue
                if pipe:
                    to = os.devnull
                else:
                    to = sys.stdout
                with stdout_redirected(to):
                    prt(append_job(f'In folder: {folder} ({i + 1}/{nfolders})',
                                   job_num=job_num))
                    if is_asr_command:
                        if defaults:
                            with set_defaults(defaults):
                                res = func.cli(args=args)
                        else:
                            res = func.cli(args=args)
                    else:
                        sys.argv = [mod.__name__] + args
                        res = func()
                if pipe:
                    click.echo(pickle.dumps(res), nl=False)
            except click.Abort:
                break
            except Exception as e:
                if not dont_raise:
                    raise
                else:
                    prt(append_job(e, job_num=job_num))
            except SystemExit:
                print('Unexpected error:', sys.exc_info()[0])
                if not dont_raise:
                    raise


@cli.command(name='list')
@click.argument('search', required=False)
def asrlist(search):
    """List and search for recipes.

    If SEARCH is specified: list only recipes containing SEARCH in their
    description.
    """
    from asr.core import get_recipes
    recipes = get_recipes()
    recipes.sort(key=lambda x: x.name)
    panel = [['Name', 'Description'],
             ['----', '-----------']]

    for recipe in recipes:
        longhelp = recipe._main.__doc__
        if not longhelp:
            longhelp = ''

        shorthelp, *_ = longhelp.split('\n')

        if search and (search not in longhelp
                       and search not in recipe.name):
            continue
        status = [recipe.name[4:], shorthelp]
        panel += [status]
    panel += ['\n']

    print(format(panel))


@cli.command()
@click.argument(
    'params', nargs=-1, type=str,
    metavar='recipe:option arg recipe:option arg'
)
def params(params: Union[str, None] = None):
    """Compile a params.json file with all options and defaults.

    This recipe compiles a list of all options and their default
    values for all recipes to be used for manually changing values
    for specific options.
    """
    from pathlib import Path
    from asr.core import get_recipes, read_json, write_json
    from ast import literal_eval
    from fnmatch import fnmatch
    import copy
    from asr.core import recursive_update

    defparamdict = {}
    recipes = get_recipes()
    for recipe in recipes:
        defparams = recipe.get_defaults()
        defparamdict[recipe.name] = defparams

    p = Path('params.json')
    if p.is_file():
        paramdict = read_json('params.json')
        # The existing valus in paramdict set the new defaults
        recursive_update(defparamdict, paramdict)
    else:
        paramdict = {}

    if isinstance(params, (list, tuple)):
        # Find recipe:option
        tmpoptions = params[::2]
        tmpargs = params[1::2]
        assert len(tmpoptions) == len(tmpargs), \
            'You must provide a value for each option'
        options = []
        args = []
        for tmpoption, tmparg in zip(tmpoptions, tmpargs):
            assert ':' in tmpoption, 'You have to use the recipe:option syntax'
            recipe, option = tmpoption.split(':')
            if '*' in recipe:
                for tmprecipe in defparamdict:
                    if not fnmatch(tmprecipe, recipe):
                        continue
                    if option in defparamdict[tmprecipe]:
                        options.append(f'{tmprecipe}:{option}')
                        args.append(tmparg)
            else:
                options.append(tmpoption)
                args.append(tmparg)

        for option, value in zip(options, args):
            recipe, option = option.split(':')

            assert option, 'You have to provide an option'
            assert recipe, 'You have to provide a recipe'

            if recipe not in paramdict:
                paramdict[recipe] = {}

            paramtype = type(defparamdict[recipe][option])
            if paramtype == dict:
                value = value.replace('...', 'None:None')
                val = literal_eval(value)
            elif paramtype == bool:
                val = literal_eval(value)
            else:
                val = paramtype(value)
            paramdict[recipe][option] = val
    elif isinstance(params, dict):
        paramdict.update(copy.deepcopy(params))
    else:
        raise NotImplementedError(
            'asr.setup.params is only compatible with'
            f'input lists and dict. Input params: {params}'
        )

    for recipe, options in paramdict.items():
        assert recipe in defparamdict, \
            f'This is an unknown recipe: {recipe}'

        for option, value in options.items():
            assert option in defparamdict[recipe], \
                f'This is an unknown option: {recipe}:{option}'
            if isinstance(value, dict):
                recursive_update(value, defparamdict[recipe][option])
                paramdict[recipe][option] = value

    if paramdict:
        write_json(p, paramdict)


@cli.group()
def cache():
    """Inspect results."""
    ...


def get_item(attrs: List[str], obj):

    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            obj = obj[attr]

    return obj


@cache.command()
@click.argument('functionname', required=False)
@click.option('-f', '--formatting',
              default=('run_specification.name '
                       'run_specification.parameters '
                       'result'), type=str)
@click.option('-s', '--sort',
              default='run_specification.name', type=str)
def ls(functionname, formatting, sort):
    from asr.core.command import full_feature_file_cache

    records = full_feature_file_cache.select()

    records = sorted(records, key=lambda x: get_item(sort.split('.'), x))
    items = formatting.split()
    formats = []
    for i, item in enumerate(items):
        item, *fmt = item.split(':')
        if fmt:
            fmt = fmt[0]
        items[i] = item
        formats.append(fmt)
    rows = [[item.split('.')[-1] for item in items]]
    for record in records:
        run_spec = record.run_specification
        if functionname and functionname != run_spec.name:
            continue
        row = []
        for item, fmt in zip(items, formats):
            obj = get_item(item.split('.'), record)
            if not fmt:
                fmt = ''
            row.append(format(obj, fmt))
        rows.append(row)

    columnwidths = [0] * len(items)
    for row in rows:
        for i, column in enumerate(row):
            columnwidths[i] = max(columnwidths[i], len(column))

    for row in rows:
        for i, column in enumerate(row):
            row[i] = column.rjust(columnwidths[i], ' ')
        print(' '.join(row))


def draw_plotly_graph(G):
    import networkx as nx
    import plotly.graph_objects as go

    pos = nx.planar_layout(G)

    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_text = []
    for node in G.nodes:
        node_text.append(str(node))

    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Dependency tree',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()


def draw_networkx_graph(G):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.layout.planar_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="C0")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=10,
        arrowstyle="->",
        arrowsize=10,
        node_color='C0',
        edge_color='C1',
        width=2,
    )

    pc = mpl.collections.PatchCollection(edges)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


@cache.command()
@click.option('--draw', is_flag=True)

def graph(draw=False):
    from asr.core.command import full_feature_file_cache

    records = full_feature_file_cache.select()

    if draw:
        import networkx as nx
        graph = nx.DiGraph()
        for record in records:
            graph.add_node(record)

        for record in records:
            for depid in record.dependencies:
                deprecord = full_feature_file_cache.get_record_from_uid(depid)
                graph.add_edge(deprecord, record)

        draw_networkx_graph(graph)
        draw_plotly_graph(graph)
    else:
        graph = {}

        for record in records:
            graph[record] = [
                full_feature_file_cache.get_record_from_uid(uid)
                for uid in record.dependencies
            ]

        count_edges_to_node = {}
        for node, edges in graph.items():
            for edge in edges:
                count = count_edges_to_node.get(edge, 0) + 1
                count_edges_to_node[edge] = count

        sorted_nodes = list(sorted(
            list(graph),
            key=lambda node: count_edges_to_node.get(node, 0),)
        )

        for node in sorted_nodes:
            print(node, '->', graph[node])


@cli.command()
@click.argument('name')
@click.option('--show/--dont-show', default=True, is_flag=True,
              help='Show generated figures')
def results(name, show):
    """Show results for a specific recipe.

    Generate and save figures relating to recipe with NAME. Examples
    of valid names are asr.bandstructure, asr.gs etc.

    """
    from matplotlib import pyplot as plt
    from asr.core import get_recipe_from_name
    from asr.core.material import (get_material_from_folder,
                                   make_panel_figures)
    recipe = get_recipe_from_name(name)

    filename = f"results-{recipe.name}.json"
    assert Path(filename).is_file(), \
        f'No results file for {recipe.name}, so I cannot show the results!'

    material = get_material_from_folder('.')
    result = material.data[filename]

    if 'ase_webpanel' not in result.get_formats():
        print(f'{recipe.name} does not have any results to present!')
        return
    from asr.database.app import create_key_descriptions
    kd = create_key_descriptions()
    panels = result.format_as('ase_webpanel', material, kd)
    print('panels', panels)
    make_panel_figures(material, panels)
    if show:
        plt.show()


@cli.command()
@click.argument('recipe')
@click.argument('hashes', required=False, nargs=-1, metavar='[HASH]...')
def find(recipe, hashes):
    """Find result files.

    Find all results files belonging to RECIPE. Optionally, filter
    these according to a certain ranges of Git hashes (requires having
    Git installed). Valid recipe names are asr.bandstructure etc.

    Find all results files calculated with a checkout of ASR that is
    an ancestor of HASH (including HASH): "asr find asr.bandstructure
    HASH".

    Find all results files calculated with a checkout of ASR that is
    an ancestor of HASH2 but not HASH1: "asr find asr.bandstructure
    HASH1..HASH2" (not including HASH1).

    Find all results files that are calculated with a checkout of ASR
    that is an ancestor of HASH1 or HASH2: "asr find asr.bandstructure
    HASH1 HASH2".

    This is basically a wrapper around Git's rev-list command and all
    hashes are forwarded to this command. For example, we can use the
    special HASH^ to refer to the parent of HASH.

    """
    from os import walk

    if not is_asr_initialized():
        initialize_asr_configuration_dir()

    recipe_results_file = f"results-{recipe}.json"

    if hashes:
        hashes = list(hashes)
        check_git()

    matching_files = []
    for root, dirs, files in walk(".", followlinks=False):

        if recipe_results_file in set(files):
            matching_files.append(str(Path(root) / recipe_results_file))

    if hashes:
        rev_list = get_git_rev_list(hashes)
        matching_files = list(
            filter(lambda x: extract_hash_from_file(x) in rev_list,
                   matching_files)
        )

    if matching_files:
        print("\n".join(matching_files))


def extract_hash_from_file(filename):
    """Extract the ASR hash from an ASR results file."""
    results = read_json(filename)
    try:
        version = results['__versions__']['asr']
    except KeyError:
        version = None
    except Exception:
        print(f"Problem when extration asr git hash from {filename}")
        raise

    if version and '-' in version:
        return version.split('-')[1]


def check_git():
    """Check that Git is installed."""
    proc = subprocess.Popen(['git'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()

    assert not err, f"{err}\nProblem with your Git installation."


def get_git_rev_list(hashes, home=None):
    """Get Git rev list from HASH1 to HASH2."""
    cfgdir = get_config_dir(home=home)

    git_repo = 'https://gitlab.com/mortengjerding/asr.git'
    if not (cfgdir / 'asr').is_dir():
        subprocess.check_output(['git', 'clone', git_repo],
                                cwd=cfgdir)

    asrdir = cfgdir / "asr"
    subprocess.check_output(['git', 'pull'],
                            cwd=asrdir)
    out = subprocess.check_output(['git', 'rev-list'] + hashes,
                                  cwd=asrdir)
    return set(out.decode("utf-8").strip("\n").split("\n"))


def is_asr_initialized(home=None):
    """Determine if ASR is initialized."""
    cfgdir = get_config_dir(home=home)
    return (cfgdir).is_dir()


def initialize_asr_configuration_dir(home=None):
    """Construct ASR configuration dir."""
    cfgdir = get_config_dir(home=home)
    cfgdir.mkdir()


def get_config_dir(home=None):
    """Get path to ASR configuration dir."""
    if home is None:
        home = Path.home()
    return home / '.asr'
