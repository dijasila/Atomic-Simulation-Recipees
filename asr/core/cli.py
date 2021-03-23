"""ASR command line interface."""
import sys
from typing import Union
import asr
from asr.core import read_json, chdir, ASRCommand
import click
from pathlib import Path
import subprocess
from ase.parallel import parprint
from functools import partial
import importlib


prt = partial(parprint, flush=True)


def format(content, indent=0, title=None, pad=2):
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
        output += out
        output += '\n'

    return output


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
@click.pass_context
def run(ctx, command, folders, not_recipe, dry_run, njobs,
        skip_if_done, dont_raise, update, must_exist):
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
                must_exist: Union[str, None] = None):
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
                prt(append_job(f'In folder: {folder} ({i + 1}/{nfolders})',
                               job_num=job_num))
                if is_asr_command:
                    func.cli(args=args)
                else:
                    sys.argv = [mod.__name__] + args
                    func()
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

    git_repo = 'https://gitlab.com/asr-dev/asr.git'
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
