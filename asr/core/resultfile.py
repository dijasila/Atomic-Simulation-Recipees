"""Functionality for converting old resultfiles to records."""

import typing
import fnmatch
import pathlib
from .dependencies import Dependency
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
    skip_patterns = [
        '*results-asr.database.fromtree.json',
        '*results-asr.database.app.json',
        '*results-asr.database.key_descriptions.json',
        '*results-asr.setup.strains*.json',
        '*displacements*/*/results-asr.database.material_fingerprint.json',
        '*strains*/results-asr.database.material_fingerprint.json',
        '*asr.setinfo*',
        '*asr.setup.params.json',
        '*asr.setup.params.json',
        '*asr.exchange@calculate.json',
    ]

    paths = []
    for path in pathlib.Path(directory).rglob('results-asr.*.json'):
        filepath = str(path)
        if any(fnmatch.fnmatch(filepath, skip_pattern)
               for skip_pattern in skip_patterns):
            continue
        paths.append(path)

    return paths


ATOMSFILES = [
    'structure.json', 'original.json', 'start.json', 'unrelaxed.json']


def construct_record_from_resultsfile(
        path: pathlib.Path,
        uids: typing.Dict[pathlib.Path, str],
):
    from .record import Record
    from asr.core.results import MetaDataNotSetError
    from asr.core import read_json, get_recipe_from_name, ASRResult
    from ase.io import read
    from asr.core.codes import Codes, Code
    folder = path.parent

    result = read_json(path)
    recipename = path.with_suffix('').name.split('-')[1]

    if isinstance(result, ASRResult):
        data = result.data
        metadata = result.metadata.todict()
    else:
        assert isinstance(result, dict)
        if recipename == 'asr.gs@calculate':
            from asr.gs import GroundStateCalculationResult
            from asr.calculators import Calculation
            calculation = Calculation(
                id='gs',
                cls_name='gpaw',
                paths=[folder / 'gs.gpw'],
            )
            result = GroundStateCalculationResult.fromdata(
                calculation=calculation)
            data = result.data
            calc = calculation.load()
            calculator = calc.parameters
            calculator['name'] = 'gpaw'
            params = {'calculator': calculator}
        else:
            data = result
            params = {}

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
        strict=False)

    try:
        parameters = result.metadata.params
        if recipename == 'asr.gs@calculate' and 'name' in parameters:
            del parameters['name']
    except MetaDataNotSetError:
        parameters = {}

    parameters = Parameters(parameters)

    atomic_structures = {
        atomsfilename: read(folder / atomsfilename).copy()
        for atomsfilename in ATOMSFILES
        if pathlib.Path(folder / atomsfilename).is_file()
    }

    params = recipe.get_argument_descriptors()
    atomsparam = [param for name, param in params.items()
                  if name == 'atoms'][0]
    atomsfilename = atomsparam['default']
    assert atomsfilename in atomic_structures

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

    name = result.metadata.asr_name

    uid = uids[path]
    dependencies = get_dependencies(path, uids)
    record = Record(
        run_specification=RunSpecification(
            name=name,
            parameters=parameters,
            version=-1,
            codes=codes,
            uid=uid,
        ),
        metadata=construct_metadata(
            directory=str(folder.absolute().relative_to(find_root()))
        ),
        resources=resources,
        result=result,
        tags=['resultfile'],
        dependencies=dependencies,
    )

    return record


def get_dependencies(path, uids):
    folder = path.parent

    deps = {
        'asr.infraredpolarizability': [
            'asr.phonons', 'asr.borncharges', 'asr.polarizability'],
        'asr.emasses@refine': [
            'asr.structureinfo', 'asr.magnetic_anisotropy', 'asr.gs'],
        'asr.emasses': [
            'asr.emasses@refine', 'asr.gs@calculate',
            'asr.gs', 'asr.structureinfo', 'asr.magnetic_anisotropy'],
        'asr.emasses@validate': ['asr.emasses'],
        'asr.berry@calculate': ['asr.gs'],
        'asr.berry': ['asr.berry@calculate'],
        'asr.gw@gs': ['asr.gs@calculate'],
        'asr.gw@gw': ['asr.gw@gs'],
        'asr.gw@empirical_mean_z': ['asr.gw@gw'],
        'asr.gw': ['asr.bandstructure', 'asr.gw@empirical_mean_z'],
        'asr.pdos@calculate': ['asr.gs'],
        'asr.pdos': ['asr.gs', 'asr.pdos@calculate'],
        'asr.phonons@calculate': [],
        'asr.phonons': ['asr.phonons@calculate'],
        'asr.push': ['asr.structureinfo', 'asr.phonons'],
        'asr.phonopy@calculate': ['asr.gs@calculate'],
        'asr.phonopy': ['asr.phonopy@calculate'],
        'asr.hse@calculate': [
            'asr.structureinfo', 'asr.gs@calculate', 'asr.gs'],
        'asr.hse': ['asr.hse@calculate', 'asr.bandstructure'],
        'asr.exchange@calculate': ['asr.gs@calculate'],
        'asr.exchange': ['asr.exchange@calculate'],
        'asr.plasmafrequency@calculate': ['asr.gs@calculate'],
        'asr.plasmafrequency': ['asr.plasmafrequency@calculate'],
        'asr.shg': ['asr.gs@calculate'],
        'asr.magstate': ['asr.gs@calculate'],
        'asr.fermisurface': ['asr.gs', 'asr.structureinfo'],
        'asr.magnetic_anisotropy': ['asr.gs@calculate', 'asr.magstate'],
        'asr.convex_hull': [
            'asr.structureinfo', 'asr.database.material_fingerprint'],
        'asr.borncharges': ['asr.gs@calculate'],
        'asr.gs': [
            'asr.gs@calculate',
            'asr.magnetic_anisotropy', 'asr.structureinfo'],
        'asr.bandstructure@calculate': ['asr.gs@calculate'],
        'asr.bandstructure': [
            'asr.bandstructure@calculate', 'asr.gs',
            'asr.structureinfo', 'asr.magnetic_anisotropy'],
        'asr.defectformation': ['asr.setup.defects', 'asr.gs'],
        'asr.deformationpotentials': ['asr.gs'],
        'asr.bader': ['asr.gs'],
        'asr.bse@calculate': ['asr.gs@calculate'],
        'asr.bse': ['asr.bse@calculate', 'asr.gs'],
        'asr.projected_bandstructure': ['asr.gs', 'asr.bandstructure'],
        'asr.shift': ['asr.gs@calculate'],
        'asr.polarizability': ['asr.structureinfo', 'asr.gs@calculate'],
    }

    name = path.with_suffix('').name.split('-')[1]

    # Some manually implemented dependencies
    if name == 'asr.piezoelectrictensor':
        dependencies = []
        dependencies += list(folder.rglob('strains*/results-asr.relax.json'))
        dependencies += list(
            folder.rglob('strains*/results-asr.formalpolarization.json')
        )
    elif name == 'asr.stiffness':
        dependencies = []
        dependencies += list(folder.rglob('strains*/results-asr.relax.json'))
    else:
        depnames = deps.get(name, [])
        dependencies = []
        for depname in depnames:
            deppath = folder / f'results-{depname}.json'
            dependencies.append(deppath)

    if dependencies:
        dep_list = []
        for dependency in dependencies:
            if dependency in uids:
                dep_list.append(Dependency(uid=uids[dependency], revision=None))
        return dep_list

    return None


def get_resultsfile_records() -> typing.List[Record]:
    records = []
    resultsfiles = find_results_files(directory=pathlib.Path('.'))
    uids = generate_uids(resultsfiles)
    for path in resultsfiles:
        try:
            record = construct_record_from_resultsfile(path, uids)
        except AssertionError as e:
            print(e)
            continue
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
        dep = [other for other in records if other.uid == dependency.uid][0]
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
    # if name == 'asr.formalpolarization':
    #     remove_keys.add('gpwname')
    # elif name == 'asr.setup.displacements':
    #     remove_keys.add('copy_params')
    # elif name in {'asr.emasses:refine', 'asr.emasses'}:
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
