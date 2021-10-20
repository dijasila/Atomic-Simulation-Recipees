"""Functionality for converting old resultfiles to records."""

import typing
import fnmatch
import pathlib
from asr.core import ASRResult
from dataclasses import dataclass
from ase import Atoms
from ase.db.core import AtomsRow
from .dependencies import Dependency, Dependencies
from .metadata import construct_metadata
from .parameters import Parameters
from .specification import RunSpecification, get_new_uuid
from .serialize import JSONSerializer
from .record import Record
from .migrate import Migration, SelectorMigrationGenerator
from .selector import Selector
from .command import get_recipes
from .utils import write_file, read_file, get_recipe_from_name
from .root import find_root


def find_directories() -> typing.List[pathlib.Path]:
    skip_patterns = [
        '*strains*',
        '*displacements*',
    ]
    directories = [pathlib.Path('.')]
    for path in pathlib.Path().rglob('*'):
        if path.is_dir() and path.name not in skip_patterns:
            directories.append(path)

    return directories


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
        '*asr.database.fromtree',
        '*asr.database.app',
        '*asr.database.key_descriptions',
        '*asr.setup.strains*',
        '*asr.database.material_fingerprint',
        '*asr.setinfo*',
        '*asr.setup.params',
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
    from .record import Record
    from asr.core.results import MetaDataNotSetError
    from asr.core.codes import Codes, Code

    result = record_context.result
    recipename = record_context.recipename
    atomic_structures = record_context.atomic_structures
    uid = record_context.uid
    dependencies = record_context.dependencies
    directory = record_context.directory

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
        if recipename == 'asr.c2db.gs@calculate' and 'name' in parameters:
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


def fix_asr_gs_record(folder, result, recipename):
    if isinstance(result, dict) and recipename == 'asr.c2db.gs@calculate':
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


def get_relevant_resultfile_parameters(path):
    from asr.core import read_json
    from ase.io import read
    folder = path.parent
    result = read_json(path)
    recipename = get_recipe_name_from_filename(path.name)
    atomic_structures = {
        atomsfilename: read(folder / atomsfilename).copy()
        for atomsfilename in ATOMSFILES
        if pathlib.Path(folder / atomsfilename).is_file()
    }
    vague_dependencies = get_vague_dependencies_from_path(path)
    directory = str(folder.absolute().relative_to(find_root()))
    return folder, result, recipename, atomic_structures, vague_dependencies, directory


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
    count = name.count(".") 
    RECIPES_THAT_WASNT_MOVED_TO_C2DB_DIRECTORY = ["asr.structureinfo", "asr.setinfo"]
    if count == 1 and name not in RECIPES_THAT_WASNT_MOVED_TO_C2DB_DIRECTORY:
        segments = name.split(".")
        name = ".".join([segments[0], "c2db", segments[1]])
    name = name.replace("@", ":")
    return name


def get_vague_dependencies_from_path(path):
    folder = path.parent
    name = get_recipe_name_from_filename(path.name)

    # Some manually implemented dependencies
    if name == 'asr.c2db.piezoelectrictensor':
        dependencies = []
        dependencies += list(folder.rglob(
            'strains*/results-asr.c2db.relax.json'))
        dependencies += list(
            folder.rglob('strains*/results-asr.c2db.formalpolarization.json')
        )
    elif name == 'asr.c2db.stiffness':
        dependencies = []
        dependencies += list(folder.rglob(
            'strains*/results-asr.c2db.relax.json'))
    else:
        depnames = get_default_dependencies_from_name(name)
        dependencies = []
        for depname in depnames:
            deppath = folder / f'results-{depname}.json'
            dependencies.append(deppath)

    if dependencies:
        return dependencies

    return None


def get_default_dependencies_from_name(name):
    deps = {
        'asr.c2db.infraredpolarizability': [
            'asr.c2db.phonons', 'asr.c2db.borncharges',
            'asr.c2db.polarizability'],
        'asr.c2db.emasses@refine': [
            'asr.structureinfo', 'asr.c2db.magnetic_anisotropy', 'asr.c2db.gs'],
        'asr.c2db.emasses': [
            'asr.c2db.emasses@refine', 'asr.c2db.gs@calculate',
            'asr.c2db.gs', 'asr.structureinfo', 'asr.c2db.magnetic_anisotropy'],
        'asr.c2db.emasses@validate': ['asr.c2db.emasses'],
        'asr.berry@calculate': ['asr.c2db.gs'],
        'asr.berry': ['asr.berry@calculate'],
        'asr.c2db.gw@gs': ['asr.c2db.gs@calculate'],
        'asr.c2db.gw@gw': ['asr.c2db.gw@gs'],
        'asr.c2db.gw@empirical_mean_z': ['asr.c2db.gw@gw'],
        'asr.c2db.gw': ['asr.c2db.bandstructure',
                        'asr.c2db.gw@empirical_mean_z'],
        'asr.c2db.pdos@calculate': ['asr.c2db.gs'],
        'asr.c2db.pdos': ['asr.c2db.gs', 'asr.c2db.pdos@calculate'],
        'asr.c2db.phonons@calculate': [],
        'asr.c2db.phonons': ['asr.c2db.phonons@calculate'],
        'asr.c2db.push': ['asr.structureinfo', 'asr.c2db.phonons'],
        'asr.c2db.phonopy@calculate': ['asr.c2db.gs@calculate'],
        'asr.c2db.phonopy': ['asr.c2db.phonopy@calculate'],
        'asr.c2db.hse@calculate': [
            'asr.structureinfo', 'asr.c2db.gs@calculate', 'asr.c2db.gs'],
        'asr.c2db.hse': ['asr.c2db.hse@calculate', 'asr.c2db.bandstructure'],
        'asr.c2db.exchange@calculate': ['asr.c2db.gs@calculate'],
        'asr.c2db.exchange': ['asr.c2db.exchange@calculate'],
        'asr.c2db.plasmafrequency@calculate': ['asr.c2db.gs@calculate'],
        'asr.c2db.plasmafrequency': ['asr.c2db.plasmafrequency@calculate'],
        'asr.c2db.shg': ['asr.c2db.gs@calculate'],
        'asr.c2db.magstate': ['asr.c2db.gs@calculate'],
        'asr.c2db.fermisurface': ['asr.c2db.gs', 'asr.structureinfo'],
        'asr.c2db.magnetic_anisotropy': ['asr.c2db.gs@calculate',
                                         'asr.c2db.magstate'],
        'asr.c2db.convex_hull': [
            'asr.structureinfo', 'asr.database.material_fingerprint'],
        'asr.c2db.borncharges': ['asr.c2db.gs@calculate'],
        'asr.c2db.gs': [
            'asr.c2db.gs@calculate',
            'asr.c2db.magnetic_anisotropy', 'asr.structureinfo'],
        'asr.c2db.bandstructure@calculate': ['asr.c2db.gs@calculate'],
        'asr.c2db.bandstructure': [
            'asr.c2db.bandstructure@calculate', 'asr.c2db.gs',
            'asr.structureinfo', 'asr.c2db.magnetic_anisotropy'],
        'asr.defectformation': ['asr.setup.defects', 'asr.c2db.gs'],
        'asr.c2db.deformationpotentials': ['asr.c2db.gs'],
        'asr.c2db.bader': ['asr.c2db.gs'],
        'asr.bse@calculate': ['asr.c2db.gs@calculate'],
        'asr.bse': ['asr.bse@calculate', 'asr.c2db.gs'],
        'asr.c2db.projected_bandstructure': ['asr.c2db.gs',
                                             'asr.c2db.bandstructure'],
        'asr.c2db.shift': ['asr.c2db.gs@calculate'],
        'asr.c2db.polarizability': ['asr.structureinfo',
                                    'asr.c2db.gs@calculate'],
    }
    return deps.get(name, [])


@dataclass
class RecordContext:

    result: typing.Any
    recipename: str
    atomic_structures: typing.Dict[str, Atoms]
    uid: str
    dependencies: typing.Optional[Dependencies]
    directory: str


def get_resultsfile_records() -> typing.List[Record]:
    contexts = get_contexts_in_current_directory()
    records = make_records_from_contexts(contexts)
    return records


def get_resultfile_records_from_database_row(row: AtomsRow):
    contexts = convert_row_data_to_contexts(row.data, row.folder)
    contexts = filter_contexts_for_unused_recipe_results(contexts)
    records = make_records_from_contexts(contexts)
    return records


def get_contexts_in_current_directory() -> typing.List[RecordContext]:
    resultsfiles = find_results_files(directory=pathlib.Path('.'))
    uids = generate_uids(resultsfiles)
    contexts = []
    for path in resultsfiles:
        try:
            (
                folder,
                result,
                recipename,
                atomic_structures,
                vague_dependencies,
                directory,
            ) = get_relevant_resultfile_parameters(path)
            uid = uids[path]
            dependencies = make_concrete_dependencies(vague_dependencies, uids)
            result = fix_asr_gs_record(folder, result, recipename)
            context = RecordContext(
                result=result,
                recipename=recipename,
                atomic_structures=atomic_structures,
                uid=uid,
                dependencies=dependencies,
                directory=directory,
            )
            contexts.append(context)
        except AssertionError as error:
            print(error)
            continue
    return contexts


def convert_row_data_to_contexts(data, directory) -> typing.List[RecordContext]:
    filenames = []
    for filename in data:
        is_results_file = filename.startswith('results-') and filename.endswith('.json')
        if not is_results_file:
            continue
        filenames.append(filename)

    uids = generate_uids(filenames)
    contexts = []
    for filename in filenames:
        result = data[filename]
        recipename = get_recipe_name_from_filename(filename)
        atomic_structures = {
            name: value for name, value in data.items()
            if name in ATOMSFILES
        }
        uid = uids[filename]
        vague_dependency_names = get_default_dependencies_from_name(recipename)
        vague_dependencies = [f"results-{dep}.json" for dep in vague_dependency_names]
        dependencies = make_concrete_dependencies(vague_dependencies, uids)
        context = RecordContext(
            result=result,
            recipename=recipename,
            atomic_structures=atomic_structures,
            uid=uid,
            dependencies=dependencies,
            directory=directory,
        )
        contexts.append(context)
    
    return contexts


def make_records_from_contexts(contexts):
    records = []
    for context in contexts:
        record = construct_record_from_context(context)
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
DEFAULTS = JSONSerializer().deserialize(read_file(PATH))


def update_resultfile_record_to_version_0(record):
    default_params = DEFAULTS[record.name]
    name = record.name
    new_parameters = Parameters({})
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

    for key in sig_parameters:
        if key in parameters:
            new_parameters[key] = parameters[key]
            unused_old_params.remove(key)
        elif key == 'atoms':
            # Atoms are treated differently
            new_parameters[key] = parameters.atomic_structures[atomsfilename]
        else:
            candidate_dependencies = []
            for depname, recipedepparams in dep_params.items():
                if key in recipedepparams:
                    candidate_dependencies.append(depname)
            if candidate_dependencies:
                assert len(candidate_dependencies) == 1
                dependency = candidate_dependencies[0]
                new_parameters[key] = dep_params[dependency][key]
                unused_dependency_params[dependency].remove(key)
            else:
                missing_params.add(key)  # new_parameters[key] = default_params[key]
    unused_old_params.remove('atomic_structures')

    # remove_keys = set(['dependency_parameters'])
    # if name == 'asr.c2db.formalpolarization':
    #     remove_keys.add('gpwname')
    # elif name == 'asr.setup.displacements':
    #     remove_keys.add('copy_params')
    # elif name in {'asr.c2db.emasses:refine', 'asr.c2db.emasses'}:
    #     remove_keys.add('gpwfilename')
    # unused_old_params = unused_old_params - remove_keys

    unused_old_params -= set(['dependency_parameters'])
    unused_dependency_params = {
        value
        for values in unused_dependency_params.values()
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

    record.run_specification.parameters = new_parameters
    record.version = 0
    return record


def get_defaults_from_all_recipes():
    defaults = {}
    for recipe in get_recipes():
        defaults[recipe.name] = recipe.defaults

    return defaults


if __name__ == '__main__':
    defaults = get_defaults_from_all_recipes()
    jsontxt = JSONSerializer().serialize(defaults)
    write_file(PATH, jsontxt)
