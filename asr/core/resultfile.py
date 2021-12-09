"""Functionality for converting old resultfiles to records."""

import fnmatch
import pathlib
import typing
from dataclasses import dataclass

from ase import Atoms
from ase.db.core import AtomsRow

from asr.core import ASRResult

from .command import get_recipes
from .dependencies import Dependencies, Dependency
from .metadata import construct_metadata
from .migrate import Migration, SelectorMigrationGenerator
from .parameters import Parameters
from .record import Record
from .results import decode_object
from .selector import Selector
from .serialize import JSONSerializer
from .specification import RunSpecification, get_new_uuid
from .utils import get_recipe_from_name, parse_mod_func, read_file, write_file


def generate_uids(resultfiles) -> typing.Dict[pathlib.Path, str]:
    return {path: get_new_uuid() for path in resultfiles}


def find_results_files(directory: pathlib.Path) -> typing.List[pathlib.Path]:
    filenames = [
        str(filename)
        for filename in pathlib.Path(directory).rglob('results-asr.*.json')
    ]
    filenames = filter_filenames_for_unused_recipe_results(filenames)
    return [pathlib.Path(filename) for filename in filenames]


def filter_filenames_for_unused_recipe_results(
    filenames
) -> typing.List[str]:
    skip_patterns = [
        '*results-asr.database.fromtree.json',
        '*results-asr.database.app.json',
        '*results-asr.database.key_descriptions.json',
        '*results-asr.setup.strains*.json',
        '*displacements*/*/results-asr.database.material_fingerprint.json',
        '*strains*/results-asr.database.material_fingerprint.json',
        '*asr.setinfo*',
        '*asr.setup.params.json',
        '*asr.c2db.exchange@calculate*',
        '*asr.c2db.exchange:calculate*',
    ]

    paths = []
    for path in filenames:
        if any(fnmatch.fnmatch(path, skip_pattern)
               for skip_pattern in skip_patterns):
            continue
        paths.append(path)

    return paths


def filter_contexts_for_unused_recipe_results(
    contexts: typing.List["RecordContext"]
) -> typing.List["RecordContext"]:
    skip_patterns = [
        '*asr.database.fromtree*',
        '*asr.database.app*',
        '*asr.database.key_descriptions*',
        '*asr.setup.strains*',
        '*asr.database.material_fingerprint*',
        '*asr.setinfo*',
        '*asr.setup.params*',
        '*asr.c2db.exchange:calculate*',
        '*asr.c2db.plasmafrequency:calculate*'
    ]
    filtered = []
    for context in contexts:
        if any(fnmatch.fnmatch(context.recipename, skip_pattern)
               for skip_pattern in skip_patterns):
            continue
        filtered.append(context)
    return filtered


ATOMSFILES = [
    'structure.json', 'original.json', 'start.json', 'unrelaxed.json']


def construct_record_from_context(
        record_context: "RecordContext",
):
    from asr.core.codes import Code, Codes
    from asr.core.results import MetaDataNotSetError

    from .record import Record

    result = record_context.result
    recipename = record_context.recipename
    atomic_structures = record_context.atomic_structures
    uid = record_context.uid
    dependencies = record_context.dependencies
    directory = record_context.directory

    mod, func = parse_mod_func(recipename)
    recipename = ':'.join([mod, func])
    if isinstance(result, ASRResult):
        data = result.data
        metadata = result.metadata.todict()
    else:
        assert isinstance(result, dict)
        data = result
        params: typing.Dict[str, typing.Any] = {}
        metadata = {
            'asr_name': recipename,
            'params': params,
        }

    recipe = get_recipe_from_name(recipename)
    if not issubclass(recipe.returns, ASRResult):
        returns = ASRResult
    else:
        returns = recipe.returns
    result = returns(
        data=data,
        metadata=metadata,
        strict=False,
    )

    try:
        parameters = result.metadata.params
        if recipename == 'asr.c2db.gs:calculate' and 'name' in parameters:
            del parameters['name']
    except MetaDataNotSetError:
        parameters = {}

    parameters = Parameters(parameters)
    params = recipe.get_argument_descriptors()
    atomsparam = [param for name, param in params.items()
                  if name == 'atoms'][0]
    atomsfilename = atomsparam['default']
    if atomsfilename not in atomic_structures:
        atomic_structures[atomsfilename] = "Unknown atoms file"

    parameters.atomic_structures = atomic_structures

    try:
        code_versions = result.metadata.code_versions
        lst_codes = []
        for package, codestr in code_versions.items():
            version, githash = codestr.split('-')
            code = Code(package, version, githash)
            lst_codes.append(code)
        codes = Codes(lst_codes)
    except MetaDataNotSetError:
        codes = Codes([])

    from asr.core.resources import Resources
    try:
        resources = result.metadata.resources
        resources = Resources(
            execution_start=resources.get('tstart'),
            execution_end=resources.get('tend'),
            execution_duration=resources['time'],
            ncores=resources['ncores'],
        )
    except MetaDataNotSetError:
        resources = Resources()

    record = Record(
        run_specification=RunSpecification(
            name=recipename,
            parameters=parameters,
            version=-1,
            codes=codes,
            uid=uid,
        ),
        metadata=construct_metadata(
            directory=directory,
        ),
        resources=resources,
        result=result,
        tags=['resultfile'],
        dependencies=dependencies,
    )

    return record


def fix_asr_gs_result_missing_calculator(folder, result, recipename):
    if isinstance(result, dict) and recipename == 'asr.c2db.gs:calculate':
        from asr.c2db.gs import GroundStateCalculationResult
        from asr.calculators import Calculation
        calculation = Calculation(
            id='gs',
            cls_name='gpaw',
            paths=[folder / 'gs.gpw'],
        )
        result = GroundStateCalculationResult.fromdata(
            calculation=calculation)
        calc = calculation.load()
        calculator = calc.parameters
        calculator['name'] = 'gpaw'
        params = {'calculator': calculator}
        metadata = {
            'asr_name': recipename,
            'params': params,
        }
        result.metadata = metadata
    return result


def fix_asr_gs_record_missing_calculator(record: "Record"):
    if record.name == 'asr.c2db.gs:calculate':
        if 'calculator' not in record.parameters:
            record.parameters.calculator = \
                OLD_DEFAULTS['asr.c2db.gs:calculate']['calculator']


def get_relevant_resultfile_parameters(path, directory):
    from ase.io import read

    from asr.core import read_json
    folder = path.parent
    result = read_json(path)
    recipename = get_recipe_name_from_filename(path.name)
    atomic_structures = {
        atomsfilename: read(folder / atomsfilename).copy()
        for atomsfilename in ATOMSFILES
        if pathlib.Path(folder / atomsfilename).is_file()
    }
    matcher = get_dependency_matcher_from_name(recipename)
    rel_directory = str(folder.absolute().relative_to(directory))
    return folder, result, recipename, atomic_structures, matcher, rel_directory


def set_context_dependencies(
    contexts: typing.List["RecordContext"],
) -> typing.List["RecordContext"]:
    for context in contexts:
        deps = []
        for context2 in contexts:
            if context.dependency_matcher(context2):
                dep = Dependency(
                    uid=context2.uid, revision=None,
                )
                deps.append(dep)
        context.dependencies = Dependencies(deps)
    return contexts


def make_concrete_dependencies(
    dependencies,
    uids
) -> typing.Optional[Dependencies]:
    if dependencies:
        dep_list = []
        for dependency in dependencies:
            if dependency in uids:
                dep_list.append(
                    Dependency(
                        uid=uids[dependency], revision=None
                    )
                )
        return Dependencies(dep_list)
    return None


def get_recipe_name_from_filename(filename):
    from os.path import splitext
    name = splitext(filename.split('-')[1])[0]
    name = name.replace("@", ":")
    name = add_main_to_name_if_missing(name)
    name = fix_recipe_name_if_recipe_has_been_moved(name)
    return name


def fix_recipe_name_if_recipe_has_been_moved(name: str) -> str:
    if is_recipe_that_was_moved_to_c2db_subpackage(name):
        name = extend_name_with_c2db_subpackage(name)
    return name


def is_recipe_that_was_moved_to_c2db_subpackage(name: str) -> bool:
    count = name.count(".")
    RECIPES_THAT_WASNT_MOVED_TO_C2DB_DIRECTORY = [
        "asr.structureinfo", "asr.setinfo",
        "asr.structureinfo:main", "asr.setinfo:main",
    ]
    if (
        name.startswith("asr")
        and count == 1
        and name not in RECIPES_THAT_WASNT_MOVED_TO_C2DB_DIRECTORY
    ):
        return True
    return False


def extend_name_with_c2db_subpackage(name: str) -> str:
    first, *rest = name.split(".")
    name = ".".join([first, "c2db", *rest])
    return name


def make_dependency_matcher(
    patterns: typing.List[str],
    attribute: str = "recipename",
) -> typing.Callable[["RecordContext"], bool]:

    def dependency_matcher(context: "RecordContext") -> bool:
        return any(
            fnmatch.fnmatch(
                str(getattr(context, attribute)),
                pattern
            )
            for pattern in patterns
        )

    return dependency_matcher


def get_dependency_matcher_from_name(
    name: str,
) -> typing.Callable[["RecordContext"], bool]:

    # Some manually implemented dependencies
    if name == 'asr.c2db.piezoelectrictensor:main':
        patterns = [
            'strains*/results-asr.c2db.relax.json',
            'strains*/results-asr.c2db.formalpolarization.json'
        ]
        return make_dependency_matcher(patterns, "path")
    elif name == 'asr.c2db.stiffness:main':
        patterns = [
            'strains*/results-asr.c2db.relax.json',
        ]
        return make_dependency_matcher(patterns, "path")

    deps : typing.Dict[str, typing.List[str]] = {
        'asr.c2db.infraredpolarizability': [
            'asr.c2db.phonons', 'asr.c2db.borncharges',
            'asr.c2db.polarizability'],
        'asr.c2db.emasses:refine': [
            'asr.structureinfo', 'asr.c2db.magnetic_anisotropy', 'asr.c2db.gs'],
        'asr.c2db.emasses': [
            'asr.c2db.emasses:refine', 'asr.c2db.gs:calculate',
            'asr.c2db.gs', 'asr.structureinfo', 'asr.c2db.magnetic_anisotropy'],
        'asr.c2db.emasses:validate': ['asr.c2db.emasses'],
        'asr.c2db.berry:calculate': ['asr.c2db.gs'],
        'asr.c2db.berry': ['asr.c2db.berry:calculate', 'asr.c2db.gs'],
        'asr.c2db.gw:gs': ['asr.c2db.gs:calculate'],
        'asr.c2db.gw:gw': ['asr.c2db.gw:gs'],
        'asr.c2db.gw:empirical_mean_z': ['asr.c2db.gw:gw'],
        'asr.c2db.gw': ['asr.c2db.bandstructure',
                        'asr.c2db.gw:empirical_mean_z'],
        'asr.c2db.pdos:calculate': ['asr.c2db.gs'],
        'asr.c2db.pdos': ['asr.c2db.gs', 'asr.c2db.pdos:calculate'],
        'asr.c2db.phonons:calculate': [],
        'asr.c2db.phonons': ['asr.c2db.phonons:calculate'],
        'asr.c2db.push': ['asr.structureinfo', 'asr.c2db.phonons'],
        'asr.c2db.phonopy:calculate': ['asr.c2db.gs:calculate'],
        'asr.c2db.phonopy': ['asr.c2db.phonopy:calculate'],
        'asr.c2db.hse:calculate': [
            'asr.structureinfo', 'asr.c2db.gs:calculate', 'asr.c2db.gs'],
        'asr.c2db.hse': ['asr.c2db.hse:calculate', 'asr.c2db.bandstructure'],
        'asr.c2db.exchange:calculate': ['asr.c2db.gs:calculate'],
        'asr.c2db.exchange': ['asr.c2db.gs:main'],
        'asr.c2db.plasmafrequency:calculate': ['asr.c2db.gs:calculate'],
        'asr.c2db.plasmafrequency': [
            'asr.c2db.plasmafrequency:calculate',
            'asr.c2db.gs',
        ],
        'asr.c2db.shg': ['asr.c2db.gs:calculate', 'asr.c2db.gs'],
        'asr.c2db.magstate': ['asr.c2db.gs:calculate'],
        'asr.c2db.fermisurface': ['asr.c2db.gs', 'asr.structureinfo'],
        'asr.c2db.magnetic_anisotropy': ['asr.c2db.gs:calculate',
                                         'asr.c2db.magstate'],
        'asr.c2db.convex_hull': [
            'asr.structureinfo', 'asr.database.material_fingerprint'],
        'asr.c2db.borncharges': ['asr.c2db.gs:calculate'],
        'asr.c2db.gs': [
            'asr.c2db.gs:calculate',
            'asr.c2db.magnetic_anisotropy', 'asr.structureinfo'],
        'asr.c2db.bandstructure:calculate': ['asr.c2db.gs:calculate'],
        'asr.c2db.bandstructure': [
            'asr.c2db.bandstructure:calculate', 'asr.c2db.gs',
            'asr.structureinfo', 'asr.c2db.magnetic_anisotropy'],
        'asr.defectformation': ['asr.setup.defects', 'asr.c2db.gs'],
        'asr.c2db.deformationpotentials': ['asr.c2db.gs'],
        'asr.c2db.bader': ['asr.c2db.gs'],
        'asr.bse:calculate': ['asr.c2db.gs:calculate'],
        'asr.bse': ['asr.bse:calculate', 'asr.c2db.gs'],
        'asr.c2db.projected_bandstructure': ['asr.c2db.gs',
                                             'asr.c2db.bandstructure'],
        'asr.c2db.shift': ['asr.c2db.gs:calculate'],
        'asr.c2db.polarizability': ['asr.structureinfo',
                                    'asr.c2db.gs:calculate'],
    }
    deps = {add_main_to_name_if_missing(key): value for key, value in deps.items()}
    dependencies = []
    for dep in deps.get(name, []):
        dep = fix_recipe_name_if_recipe_has_been_moved(
            add_main_to_name_if_missing(dep))
        dependencies.append(dep)

    return make_dependency_matcher(dependencies, "recipename")


def add_main_to_name_if_missing(dep: str) -> str:
    if ':' not in dep:
        dep = dep + ':main'
    return dep


@dataclass
class RecordContext:
    """Class that contain the contextual data to create a record."""

    result: typing.Any
    recipename: str
    atomic_structures: typing.Dict[str, Atoms]
    uid: str
    dependency_matcher: typing.Callable[["RecordContext"], bool]
    dependencies: typing.Optional[Dependencies]
    directory: str
    path: typing.Optional[pathlib.Path] = None


def get_resultfile_records_in_directory(
    directory: pathlib.Path
) -> typing.List[Record]:
    contexts = get_contexts_in_directory(directory)
    contexts = filter_contexts_for_unused_recipe_results(contexts)
    contexts = set_context_dependencies(contexts)
    records = make_records_from_contexts(contexts)
    return records


def get_resultfile_records_from_database_row(row: AtomsRow):
    contexts = convert_row_data_to_contexts(row.data, row.folder)
    contexts = filter_contexts_for_unused_recipe_results(contexts)
    contexts = set_context_dependencies(contexts)
    records = make_records_from_contexts(contexts)
    return records


def get_contexts_in_directory(
    directory=pathlib.Path('.')
) -> typing.List[RecordContext]:
    resultsfiles = find_results_files(directory=directory)
    uids = generate_uids(resultsfiles)
    contexts = []
    for path in resultsfiles:
        try:
            (
                folder,
                result,
                recipename,
                atomic_structures,
                matcher,
                rel_directory,
            ) = get_relevant_resultfile_parameters(path, directory)
        except ModuleNotFoundError as error:
            print(error)
            continue
        uid = uids[path]
        result = fix_asr_gs_result_missing_calculator(folder, result, recipename)
        context = RecordContext(
            result=result,
            recipename=recipename,
            atomic_structures=atomic_structures,
            uid=uid,
            dependency_matcher=matcher,
            dependencies=None,
            directory=rel_directory,
            path=path,
        )
        contexts.append(context)
    return contexts


def convert_row_data_to_contexts(data, directory) -> typing.List[RecordContext]:
    filenames = []
    for filename in data:
        is_results_file = filename.startswith('results-') and filename.endswith('.json')
        if not is_results_file:
            continue
        filenames.append(filename)

    contexts = []

    children_data = data.get('__children_data__', {})
    for child_values in children_data.values():
        directory = child_values['directory']
        child_data = child_values['data']
        child_contexts = convert_row_data_to_contexts(child_data, directory)
        contexts.extend(child_contexts)

    uids = generate_uids(filenames)
    for filename in filenames:
        recipename = get_recipe_name_from_filename(filename)
        result = data[filename]

        try:
            result = decode_object(result)
        except ModuleNotFoundError as err:
            print(err)
            continue
        result = fix_result_object_missing_name(recipename, result)

        path = pathlib.Path(directory) / \
            f'results-{remove_main_in_name(recipename)}.json'
        atomic_structures = {}
        for name, value in data.items():
            if name in ATOMSFILES:
                dct = value['1']
                atoms = Atoms(
                    numbers=dct.get("numbers"),
                    positions=dct.get("positions"),
                    cell=dct.get("cell"),
                    pbc=dct.get("pbc"),
                    magmoms=dct.get('initial_magmoms'),
                    charges=dct.get('initial_charges'),
                    tags=dct.get('tags'),
                    masses=dct.get('masses'),
                    momenta=dct.get('momenta'),
                    constraint=dct.get("constraints"),
                )
                atomic_structures[name] = atoms

        uid = uids[filename]
        matcher = get_dependency_matcher_from_name(recipename)
        context = RecordContext(
            result=result,
            recipename=recipename,
            atomic_structures=atomic_structures,
            uid=uid,
            dependency_matcher=matcher,
            dependencies=None,
            directory=directory,
            path=path,
        )
        contexts.append(context)

    return contexts


def fix_result_object_missing_name(recipename, result):
    if isinstance(result, dict) and '__asr_name__' not in result:
        result['__asr_name__'] = recipename
        result = decode_object(result)
    return result


def remove_main_in_name(name: str) -> str:
    if name.endswith(':main'):
        name = name[:-5]
    return name


def make_records_from_contexts(contexts):
    records = []
    for context in contexts:
        record = construct_record_from_context(context)
        fix_asr_gs_record_missing_calculator(record)
        records.append(record)
    records = inherit_dependency_parameters(records)
    return records


def inherit_dependency_parameters(records):

    for record in records:
        deps = record.dependencies or []  # [record.uid for record in records]
        dep_params = get_dependency_parameters(deps, records)
        dep_params = Parameters({
            'dependency_parameters': Parameters(dep_params)})
        record.parameters.update(dep_params)

    return records


def get_dependency_parameters(dependencies, records):
    params = Parameters({})
    if dependencies is None:
        return params

    for dependency in dependencies:
        deps = [other for other in records if other.uid == dependency.uid]
        if not deps:
            continue
        dep = deps[0]
        depparams = Parameters(
            {
                dep.name: {
                    key: value for key, value in dep.parameters.items()
                    if key != 'dependency_parameters'
                }
            }
        )
        params.update(depparams)
        params.update(get_dependency_parameters(dep.dependencies, records))

    return params


def get_resultfile_migration_generator() -> SelectorMigrationGenerator:

    mig = Migration(
        update_resultfile_record_to_version_0,
        uid='9269242a035a4731bcd5ac609ff0a086',
        description='Extract missing parameters from dependencies, '
        'add those to parameters, '
        'and increase version to 0.',
        eagerness=-1,
    )

    sel = Selector()
    sel.version = sel.EQ(-1)

    make_migrations = SelectorMigrationGenerator(
        selector=sel, migration=mig)
    return make_migrations


PATH = pathlib.Path(__file__).parent / 'old_resultfile_defaults.json'
TMP_DEFAULTS = JSONSerializer().deserialize(read_file(PATH))
OLD_DEFAULTS: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
for tmpkey, value in TMP_DEFAULTS.items():
    OLD_DEFAULTS[
        add_main_to_name_if_missing(
            fix_recipe_name_if_recipe_has_been_moved(tmpkey)
        )
    ] = value


def update_resultfile_record_to_version_0(record):
    default_params = OLD_DEFAULTS[record.name]
    name = record.name
    recipe = get_recipe_from_name(name)
    sig = recipe.get_signature()
    parameters = record.parameters

    sig_parameters = {parameter for parameter in sig.parameters}

    unused_old_params = set(parameters.keys())
    missing_params = set()
    dep_params = record.parameters.get('dependency_parameters', {})
    unused_dependency_params = {name: set(values)
                                for name, values in dep_params.items()}

    params = recipe.get_argument_descriptors()
    atomsparam = [param for name, param in params.items()
                  if name == 'atoms'][0]
    atomsfilename = atomsparam['default']

    new_parameters = Parameters({})
    for key in sig_parameters:
        if key in parameters:
            param_value = parameters[key]
            new_parameters[key] = param_value
            unused_old_params.remove(key)
            remove_matching_dependency_params(
                dep_params, unused_dependency_params, key, param_value)
            continue
        elif key == 'atoms':
            # Atoms are treated differently
            new_parameters[key] = parameters.atomic_structures[atomsfilename]
            continue
        dep_names, dep_values = find_deps_matching_key(
            dep_params, key,
        )
        if dep_values:
            assert all_values_equal(dep_values)
            new_parameters[key] = dep_values[0]
            for dependency in dep_names:
                remove_dependency_param(unused_dependency_params, key, dependency)
        else:
            missing_params.add(key)

    unused_old_params.remove('atomic_structures')
    unused_old_params -= set(['dependency_parameters'])
    unused_dependency_params = {
        (name, value)
        for name, values in unused_dependency_params.items()
        for value in values
        if value != 'atomic_structures'
    }

    if missing_params and not unused_old_params and not unused_dependency_params:
        for key in missing_params:
            new_parameters[key] = default_params[key]
    else:
        missing_params_msg = (
            'Could not extract following parameters from '
            f'dependencies={missing_params}. '
        )
        unused_old_params_msg = \
            f'Parameters from resultfile record not used={unused_old_params}. '
        unused_dependency_params_msg = \
            f'Dependency parameters unused={unused_dependency_params}. '
        assert not (unused_old_params or missing_params or unused_dependency_params), (
            ''.join(
                [
                    missing_params_msg if missing_params else '',
                    unused_old_params_msg if unused_old_params else '',
                    unused_dependency_params_msg if unused_dependency_params else '',
                    f'Please add a migration for {name} that fixes these issues '
                    'and run migration tool again.',
                ]
            )
        )

    # Sanity check
    assert set(new_parameters.keys()) == set(sig_parameters)

    record.run_specification.parameters = new_parameters
    record.version = 0
    return record


def remove_matching_dependency_params(
    dep_params,
    unused_dependency_params,
    key,
    param_value,
):
    dep_names, dep_values = find_deps_matching_key(dep_params, key)
    for dep_name, dep_value in zip(dep_names, dep_values):
        if dep_value == param_value:
            remove_dependency_param(
                unused_dependency_params, key, dep_name)


def all_values_equal(values):
    first_value = values[0]
    return all(
        first_value == other_value
        for other_value in values[1:]
    )


def find_deps_matching_key(dep_params, key):
    candidate_dependency_names = []
    candidate_dependency_values = []
    for depname, recipedepparams in dep_params.items():
        try:
            candidate_dependency_values.append(recipedepparams[key])
        except KeyError:
            continue
        candidate_dependency_names.append(depname)
    return candidate_dependency_names, candidate_dependency_values


def get_defaults_from_all_recipes():
    defaults = {}
    for recipe in get_recipes():
        defaults[recipe.name] = recipe.defaults

    return defaults


def remove_dependency_param(unused_dependency_params, key, dependency):
    unused_dependency_params[dependency].remove(key)


if __name__ == '__main__':
    defaults = get_defaults_from_all_recipes()
    jsontxt = JSONSerializer().serialize(defaults)
    write_file(PATH, jsontxt)
