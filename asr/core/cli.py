"""ASR command line interface."""
from ast import literal_eval
from contextlib import contextmanager
from functools import partial
import importlib
import os
from pathlib import Path
import pickle
import sys
import traceback
from typing import Union, Dict, Any, List, Tuple
import asr
from asr.core import (
    chdir, ASRCommand, DictStr, set_defaults, get_cache, CommaStr,
    get_recipes,
)
import click
from ase.parallel import parprint

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


def format_list(content, indent=0, title=None, pad=2):
    colwidth_c = []
    for row in content:
        if isinstance(row, str):
            continue
        for c, element in enumerate(row):
            nchar = len(element)
            try:
                colwidth_c[c] = max(colwidth_c[c], nchar)
            except IndexError:
                colwidth_c.append(nchar)

    output = ''
    if title:
        output = f'\n{title}\n'
    for row in content:
        out = ' ' * indent
        if isinstance(row, str):
            output += f'{row}'
            continue
        for colw, desc in zip(colwidth_c, row):
            out += f'{desc: <{colw}}' + ' ' * pad
        output += out.rstrip()
        output += '\n'

    return output.rstrip()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=asr.__version__)
def cli():
    ...


@cli.command()
@click.argument("directories", nargs=-1,
                type=click.Path(resolve_path=True),
                metavar='[directory]')
def init(directories):
    """Initialize ASR Repository.

    Initialize asr repository in directory. Defaults to '.' if no
    directory is supplied.

    """
    from .root import initialize_root
    if not directories:
        directories = [Path('.')]
    for directory in directories:
        initialize_root(directory)


@cli.command()
@click.argument('command', nargs=1)
@click.argument('folders', nargs=-1)
@click.option('-n', '--not-recipe', is_flag=True,
              help='COMMAND is not a recipe.')
@click.option('-z', '--dry-run', is_flag=True,
              help='Show what would happen without doing anything.')
@click.option(
    '-j', '--njobs', type=int, default=1,
    help='Run COMMAND in parallel on JOBS processes distributed over FOLDERS.')
@click.option('--dont-raise', is_flag=True, default=False,
              help='Continue to next folder when encountering error.')
@click.option('--must-exist', type=str,
              help="Skip folder where this file doesn't exist.")
@click.option('--defaults', type=DictStr(),
              help="Set default parameters. Takes precedence over params.json.")
@click.pass_context
def run(ctx, command, folders, not_recipe, dry_run, njobs,
        dont_raise, must_exist,
        defaults):
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

    kwargs = {
        'dont_raise': dont_raise,
        'dry_run': dry_run,
        'not_recipe': not_recipe,
        'command': command,
        'must_exist': must_exist,
        'defaults': defaults,
    }
    if njobs > 1:
        prt(f'Number of jobs: {njobs}')
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
                dont_raise: bool,
                job_num: Union[int, None] = None,
                must_exist: Union[str, None] = None,
                defaults: Dict[str, Any]):
    """Run command in folders."""
    nfolders = len(folders)
    module, *args = command.split()
    function = None
    if ':' in module:
        module, function = module.split(':')

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
        append_job(f'{module}:{function} doesn\'t exist.', job_num=job_num)
    func = getattr(mod, function)

    if isinstance(func, ASRCommand):
        is_asr_command = True
    else:
        is_asr_command = False

    if dry_run:
        prt(append_job(f'Would run {module}:{function} '
                       f'in {nfolders} folders.', job_num=job_num))
        return

    for i, folder in enumerate(folders):
        with chdir(Path(folder)):
            try:
                if must_exist and not Path(must_exist).exists():
                    continue
                pipe = not sys.stdout.isatty()
                if pipe:
                    to = sys.stderr
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
    recipes = get_recipes()
    recipes.sort(key=lambda x: x.name)
    panel = [['Name', 'Description'],
             ['----', '-----------']]

    for recipe in recipes:
        longhelp = recipe.get_wrapped_function().__doc__
        if not longhelp:
            longhelp = ''

        shorthelp, *_ = longhelp.split('\n')

        if search and (search not in longhelp
                       and search not in recipe.name):
            continue

        assert recipe.name.startswith('asr.')
        name = recipe.name
        status = [name, shorthelp]
        panel += [status]

    print(format_list(panel))


def recipes_as_dict():
    return {recipe.name: recipe for recipe in get_recipes()}


@cli.command()
@click.argument('recipe', nargs=1)
@click.argument(
    'params', nargs=-1, type=str, required=True,
    metavar='OPTION=VALUE...',
)
def params(recipe, params: Union[str, None] = None):
    """Compile a params.json file with all options and defaults.

    This recipe compiles a list of all options and their default
    values for all recipes to be used for manually changing values
    for specific options.
    """
    return _params(recipe, params)


def _params(name: str, params: str):
    from collections.abc import Sequence
    from asr.core import read_json, write_json
    import copy
    from asr.core import recursive_update

    all_recipes = recipes_as_dict()
    defparamdict = {recipe.name: recipe.defaults
                    for recipe in all_recipes.values()}

    recipe = all_recipes[name]

    params_path = Path('params.json')
    if params_path.is_file():
        paramdict = read_json(params_path)
        recursive_update(defparamdict, paramdict)
    else:
        paramdict = {}

    if isinstance(params, Sequence):
        # XXX if '*' in recipe:
        # XXX     for tmprecipe in defparamdict:
        # XXX         if not fnmatch(tmprecipe, recipe):
        # XXX             continue
        # XXX         if option in defparamdict[tmprecipe]:
        # XXX             options.append(f'{tmprecipe}:{option}')
        # XXX             args.append(tmparg)

        paramdict.setdefault(name, {})

        for directive in params:
            keyword, value = directive.split('=', 1)

            mydefaults = defparamdict[recipe.name]
            paramtype = type(mydefaults[keyword])
            if paramtype == dict:
                value = value.replace('...', 'None:None')
                val = literal_eval(value)
            elif paramtype == bool:
                val = literal_eval(value)
            else:
                val = paramtype(value)
            paramdict[name][keyword] = val
    elif isinstance(params, dict):
        paramdict.update(copy.deepcopy(params))
    else:
        raise NotImplementedError(
            'asr.setup.params is only compatible with'
            f'input lists and dict. Input params: {params}'
        )

    for name, options in paramdict.items():
        for option, value in options.items():
            assert option in defparamdict[name], \
                f'This is an unknown option: {name}:{option}'
            if isinstance(value, dict):
                recursive_update(value, defparamdict[name][option])
                paramdict[name][option] = value

    if paramdict:
        write_json(params_path, paramdict)


@cli.group()
def cache():
    """Inspect results."""


def get_item(attrs: List[str], obj):

    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            try:
                obj = obj[attr]
            except TypeError:
                obj = None

    return obj


@cache.command()
@click.argument("directories", nargs=-1,
                type=click.Path(resolve_path=True),
                metavar='[directory]')
def add_resultfile_records(directories):
    """Find legacy "results" files and store them as Records.

    Search directories (or working directory if not given) for legacy results
    file, adding a Record to the cache for each file.
    """
    from asr.core.resultfile import get_resultsfile_records
    from .utils import chdir
    if not directories:
        directories = [Path('.').resolve()]
    for directory in directories:
        with chdir(directory):
            print(directory)
            cache = get_cache()
            resultfile_records = get_resultsfile_records()

            records_to_add = []
            for record in resultfile_records:
                if not cache.has(name=record.name,
                                 version=record.version,
                                 parameters=record.parameters):
                    records_to_add.append(record)

            for record in records_to_add:
                print(f'Adding resultfile {record.name} to cache.')
                cache.add(record)


@cache.command()
@click.argument('selection', required=False, nargs=-1)
@click.option('-a', '--apply', is_flag=True, help='Apply migrations.')
@click.option('-v', '--verbose', is_flag=True, help='Apply migrations.')
@click.option('-e', '--show-errors', is_flag=True,
              help='Show tracebacks for migration errors.')
def migrate(selection, apply=False, verbose=False, show_errors=False):
    """Look for cache migrations."""
    from asr.core.migrate import (
        migrate_record,
        get_migration_generator,
    )

    cache = get_cache()
    sel = make_selector_from_selection(cache, selection)
    make_migrations = get_migration_generator()
    record_migrations = []
    erroneous_migrations = []
    nup_to_date = 0
    nmigrations = 0
    nerrors = 0

    for record in cache.select(selector=sel):
        record_migration = migrate_record(record, make_migrations)
        if record_migration:
            nmigrations += 1
            record_migrations.append(record_migration)

        if record_migration.has_errors():
            nerrors += 1
            erroneous_migrations.append(record_migration)

        if not (record_migration
                or record_migration.has_errors()):
            nup_to_date += 1

    if nmigrations == 0 and nerrors == 0:
        print('All records up to date. No migrations to apply.')
        return

    if verbose:
        nmigrations = len(record_migrations)
        strs = []
        for i, migration in enumerate(record_migrations):
            strs.append(f'#{i} {migration}')
        print('\n\n'.join(strs))
        print()

    if show_errors:
        print('Showing errors for migrations:')
        for record_migration in erroneous_migrations:
            print(f'Error for: {record_migration}')
            for migration, error in record_migration.errors:
                print(f'Error in: {migration}')
                traceback.print_exception(
                    type(error), error, error.__traceback__,
                )
                print()

    print(
        '\n'.join(
            [
                f'There are {nmigrations} unapplied migrations, '
                f'{nerrors} erroneous migrations and '
                f'{nup_to_date} records are up to date.',
                '',
            ]
        )
    )

    if not apply and nmigrations > 0:
        print(
            '\n'.join(
                [
                    'Run',
                    '    $ asr cache migrate --apply',
                    'to apply migrations.',
                ]
            )
        )

    if apply:
        for record_migration in record_migrations:
            print(record_migration)
            print()
            record_migration.apply(cache)


@cache.command()
def new_uid():
    from .specification import get_new_uuid

    print(get_new_uuid())


def make_selector_from_selection(cache, selection):
    selector = cache.make_selector()
    if selection:
        for keyvalue in selection:
            key, value = keyvalue.split('=')
            try:
                value = float(value)
            except ValueError:
                pass
            setattr(selector, key, selector.EQ(value))
    return selector


@cache.command()
@click.argument('selection', required=False, nargs=-1)
@click.option('-f', '--formatting',
              default=('run_specification.name '
                       'run_specification.parameters '
                       'result '
                       ), type=str)
@click.option('-s', '--sort',
              default='run_specification.name', type=str)
@click.option('-w', '--width', default=40, type=int,
              help='Maximum width of column.')
@click.option('-i', '--include-migrated', is_flag=True,
              help='Also include migrated records.')
def ls(selection, formatting, sort, width, include_migrated):
    """List records in cache."""
    cache = get_cache()
    selector = make_selector_from_selection(cache, selection)

    records = cache.select(selector=selector)
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
        row = []
        for item, fmt in zip(items, formats):
            if item == 'record':
                obj = record
            else:
                obj = get_item(item.split('.'), record)
            if not fmt:
                fmt = ''
            text = format(obj, fmt)
            if len(text) > width:
                text = text[:width] + '...'
            row.append(text)
        rows.append(row)

    columnwidths = [0] * len(items)
    for row in rows:
        for i, column in enumerate(row):
            columnwidths[i] = max(columnwidths[i], len(column))

    for row in rows:
        for i, column in enumerate(row):
            row[i] = column.rjust(columnwidths[i], ' ')
        print(' '.join(row))


@cache.command()
@click.argument('selection', required=False, nargs=-1)
@click.option('-z', '--dry-run', is_flag=True,
              help='Print what will happen without doing anything.')
def rm(selection, dry_run):
    """Remove records from cache."""
    cache = get_cache()
    selector = make_selector_from_selection(cache, selection)

    if dry_run:
        records = cache.select(selector=selector)
    else:
        records = cache.remove(selector=selector)

    for i, record in enumerate(records):
        print(f'#{i} {record.run_specification.name}')

    if dry_run:
        print(f'Would delete {len(records)} record(s).')
    else:
        print(f'Deleted {len(records)} record(s).')


@cache.command()
@click.argument('selection', required=False, nargs=-1)
def detail(selection):
    """Detail records."""
    cache = get_cache()
    selector = make_selector_from_selection(cache, selection)

    records = cache.select(selector=selector)
    for record in records:
        print(str(record))


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


def draw_networkx_graph(G, labels=False, saveto=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.layout.planar_layout(G)

    if labels:
        lab = {node: node.name for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=lab,
                                verticalalignment='bottom')
    else:
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
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="C0")
    mpl.collections.PatchCollection(edges)

    ax = plt.gca()
    ax.set_axis_off()
    plt.tight_layout()
    if saveto:
        plt.savefig(saveto)
    plt.show()


@cache.command()
@click.option('--draw', is_flag=True)
@click.option('--labels', is_flag=True)
@click.option('--saveto', help='Save to filename')
def graph(draw=False, labels=False, saveto=None):
    """Show graph of cached data [unfinished]."""
    cache = get_cache()
    records = cache.select()

    if draw:
        import networkx as nx
        graph = nx.DiGraph()
        for record in records:
            graph.add_node(record, label=record.name)

        for record in records:
            for depid in record.dependencies:
                deprecord = cache.get(
                    run_specification=dict(uid=depid))
                graph.add_edge(deprecord, record)

        draw_networkx_graph(graph, labels=labels, saveto=saveto)
        # draw_plotly_graph(graph)
    else:
        graph = {}

        for record in records:
            graph[record] = [
                cache.get(run_specification=dict(uid=uid))
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
            print(node, '<-', graph[node])


@cli.command()
@click.argument('selection', required=False, nargs=-1)
@click.option('--show/--dont-show', default=True, is_flag=True,
              help='Show generated figures')
def results(selection, show):
    """Show results from records.

    Generate and save figures relating to recipe with NAME. Examples
    of valid names are asr.bandstructure, asr.gs etc.

    """
    from matplotlib import pyplot as plt
    from asr.core.material import (get_row_from_folder,
                                   make_panel_figures)
    cache = get_cache()
    selector = make_selector_from_selection(cache, selection)
    records = cache.select(selector=selector)

    assert records, 'No matching records!'

    from asr.database.app import create_key_descriptions
    kd = create_key_descriptions()

    for record in records:
        result = record.result
        if 'ase_webpanel' not in result.get_formats():
            print(f'{result} does not have any results to present!')
            continue
        row = get_row_from_folder('.')
        panels = result.format_as('ase_webpanel', row, kd)
        make_panel_figures(row, panels, uid=record.uid[:10])

        print('panels', panels)

    if show:
        plt.show()


@cli.group()
def database():
    """ASR material project database."""
    pass


@database.command()
@click.argument('folders', nargs=-1, type=str)
@click.option('-r', '--recursive', is_flag=True,
              help='Recurse and collect subdirectories.')
@click.option('--children-patterns', type=str, default='')
@click.option('--patterns', help='Only select files matching pattern.', type=str,
              default='info.json,params.json')
@click.option('--dbname', help='Database name.', type=str, default='database.db')
@click.option('--njobs', type=int, default=1,
              help='Delegate collection of database to NJOBS subprocesses. '
              'Can significantly speed up database collection.')
def fromtree(
        folders: Union[str, None],
        recursive: bool,
        children_patterns: str,
        patterns: str,
        dbname: str,
        njobs: int,
):
    """Generate database from directory tree."""
    from asr.database.fromtree import main

    main(folders=folders, recursive=recursive,
         children_patterns=children_patterns,
         patterns=patterns,
         dbname=dbname,
         njobs=njobs)


@database.command()
@click.argument('database', nargs=1, type=str)
@click.option('--run/--dry-run', is_flag=True)
@click.option('-s', '--selection', help='ASE-DB selection', type=str,
              default='')
@click.option('-t', '--tree-structure', type=str,
              default='tree/{stoi}/{reduced_formula:abc}/{row.uid}')
@click.option('--sort', help='Sort the generated materials '
              '(only useful when dividing chunking tree)', type=str)
@click.option(
    '--copy/--no-copy', is_flag=True, help='Copy pointer tagged files')
@click.option('--atomsfile',
              help="Filename to unpack atomic structure to. "
              "By default, don't write atoms file.",
              type=str)
@click.option(
    '-c', '--chunks', metavar='N', help='Divide the tree into N chunks',
    type=int, default=1)
@click.option(
    '--patterns',
    help="Comma separated patterns. Only unpack files matching patterns",
    type=str,
    default='*')
@click.option('--update-tree', is_flag=True,
              help='Update results files in existing folder tree.')
def main(database: str, run: bool, selection: str,
         tree_structure: str,
         sort: str,
         atomsfile: str,
         chunks: int,
         copy: bool,
         patterns: str,
         update_tree: bool):
    """Unpack database into directory tree."""
    from asr.database.totree import main as totree

    totree(
        database=database,
        run=run,
        selection=selection,
        tree_structure=tree_structure,
        sort=sort,
        atomsfile=atomsfile,
        chunks=chunks,
        copy=copy,
        patterns=patterns,
        update_tree=update_tree,
    )


@database.command()
@click.argument("databases", nargs=-1, type=str)
@click.option("--host", help="Host address.", type=str, default='0.0.0.0')
@click.option("--test", is_flag=True, help="Test the app.")
@click.option("--extra_kvp_descriptions", type=str,
              help='File containing extra kvp descriptions for info.json')
def app(databases, host, test, extra_kvp_descriptions):
    """Run the database web app."""
    from asr.database.app import main
    main(databases=databases, host=host, test=test,
         extra_kvp_descriptions=extra_kvp_descriptions)


@database.command()
@click.option('--target', type=str,
              help='Target DB you want to create the links in.')
@click.argument('dbs', nargs=-1, type=str)
def crosslinks(target: str,
               dbs: Union[str, None] = None):
    from asr.database.crosslinks import main
    main(target=target, dbs=dbs)


@database.command()
@click.option('--include', help='Comma-separated string of folders to include.',
              type=CommaStr())
@click.option('--exclude', help='Comma-separated string of folders to exclude.',
              type=CommaStr())
def treelinks(include: str = '',
              exclude: str = ''):
    from asr.database.treelinks import main
    main(include=include, exclude=exclude)


class KeyValuePair(click.ParamType):
    """Read atoms object from filename and return Atoms object."""

    def convert(self, value, param, ctx):
        """Convert string to a (key, value) tuple."""
        assert ':' in value
        key, value = value.split(':')
        if not value == '':
            value = literal_eval(value)
        return key, value


@database.command()
@click.argument('key_value_pairs', metavar='key:value', nargs=-1,
                type=KeyValuePair())
def setinfo(key_value_pairs: List[Tuple[str, str]]):
    """Set additional key value pairs.

    These extra key value pairs are stored in info.json.  To set a key
    value pair simply do::

        asr database setinfo key1:'mystr' key2:1 key3:True

    The values supplied values will be interpred and the result will
    be {'key1': 'mystr', 'key2': 1, 'key3': True}

    Some key value pairs are protected and can assume a limited set of
    values::

        - `first_class_material`: True, False

    To delete an existing key-value-pair in info.json supply an empty
    string as a value, i.e.:

    asr database setinfo mykey:

    would delete "mykey".

    """
    from asr.setinfo import main

    main(key_value_pairs)
