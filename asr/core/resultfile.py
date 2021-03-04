"""Functionality for converting old resultfiles to records."""

import typing
import fnmatch
import pathlib
from .parameters import Parameters
from .specification import RunSpecification, get_new_uuid
from .serialize import JSONSerializer
from .record import Record
from .migrate import RecordMutation
from .selector import Selector
from .command import get_recipes
from .utils import write_file, read_file


def find_directories() -> typing.List[pathlib.Path]:
    directories = []
    for path in pathlib.Path().rglob('*'):
        if path.is_dir():
            directories.append(path)

    return directories


def generate_uids(resultfiles) -> typing.Dict[pathlib.Path, str]:
    return {path: get_new_uuid() for path in resultfiles}


def find_results_files() -> typing.List[pathlib.Path]:
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
    ]

    paths = []
    for path in pathlib.Path().rglob('results-asr.*.json'):
        filepath = str(path)
        if any(fnmatch.fnmatch(filepath, skip_pattern)
               for skip_pattern in skip_patterns):
            continue
        paths.append(path)

    return paths


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
    atoms = read(folder / 'structure.json')

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
            metadata = {
                'asr_name': recipename,
                'params': {'calculator': calculator},
            }
        else:
            raise AssertionError(f'Unparsable old results file: path={path}')

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
    except MetaDataNotSetError:
        parameters = {}

    parameters = Parameters(parameters)
    if 'atoms' not in parameters:
        parameters.atoms = atoms.copy()

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
    if '@' not in name:
        name += ':main'
    else:
        name = name.replace('@', ':')

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
        'asr.phonons@calculate': ['asr.gs@calculate'],
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
                dep_list.append(uids[dependency])
        return dep_list

    return None


def get_resultsfile_records() -> typing.List[Record]:
    resultsfiles = find_results_files()
    records = []
    uids = generate_uids(resultsfiles)
    for path in resultsfiles:
        record = construct_record_from_resultsfile(path, uids)
        records.append(record)

    records = inherit_dependency_parameters(records)
    return records


def inherit_dependency_parameters(records):

    for record in records:
        dep_uids = get_dependent_uids(record, records)
        dep_params = get_dependency_parameters(dep_uids, records)
        dep_params = Parameters({'dependency_parameters': dep_params})
        record.parameters.update(dep_params)

    return records


def get_dependent_uids(record, records):

    if record.dependencies is None:
        return set()
    dependent_uids = set(record.dependencies)

    uid_list = []
    for dep_uid in dependent_uids:
        dep_record = [other for other in records if other.uid == dep_uid][0]
        uids = get_dependent_uids(dep_record, records)
        uid_list.append(uids)

    for uids in uid_list:
        dependent_uids.update(uids)

    return dependent_uids


def get_dependency_parameters(dependency_uids, records):
    params = Parameters({})
    if dependency_uids is None:
        return params

    for dependency in dependency_uids:
        dep = [other for other in records if other.uid == dependency][0]
        depparams = Parameters(
            {
                dep.name: {
                    key: value for key, value in dep.parameters.items()
                    if key != 'dependency_parameters'
                }
            }
        )
        params.update(depparams)

    return params


def get_resultfile_mutations() -> typing.List[RecordMutation]:
    sel = Selector()
    sel.tags = sel.CONTAINS('resultfile')
    sel.version = sel.EQ(-1)
    return [
        RecordMutation(
            add_default_parameters,
            uid='9269242a035a4731bcd5ac609ff0a086',
            selector=sel,
            description='Add missing parameters to record from resultfile.',
        )
    ]


PATH = pathlib.Path(__file__).parent / 'old_resultfile_defaults.json'
DEFAULTS = JSONSerializer().deserialize(read_file(PATH))


def add_default_parameters(record):
    from .utils import get_recipe_from_name
    default_params = DEFAULTS[record.name]
    name = record.name
    new_parameters = Parameters({})
    recipe = get_recipe_from_name(name)
    sig = recipe.get_signature()
    parameters = record.parameters

    sig_parameters = {parameter for parameter in sig.parameters}

    unknown_keys = set()
    for key, value in record.parameters.items():
        if key not in sig_parameters:
            unknown_keys.add(key)
        else:
            new_parameters[key] = value

    unused_old_params = set(parameters.keys())
    for key in sig_parameters:
        if key in parameters:
            new_parameters[key] = parameters[key]
            unused_old_params.remove(key)
        else:
            dep_params = record.parameters.get('dependency_parameters', {})
            candidate_params = []
            for depname, recipedepparams in dep_params.items():
                if key in recipedepparams:
                    candidate_params.append(recipedepparams[key])
            if candidate_params:
                assert len(candidate_params) == 1
                new_parameters[key] = candidate_params[0]
            else:
                new_parameters[key] = default_params[key]

    remove_keys = set(['dependency_parameters'])
    if name == 'asr.formalpolarization:main':
        remove_keys.add('gpwname')
    elif name == 'asr.setup.displacements:main':
        remove_keys.add('copy_params')
    elif name in {'asr.emasses:refine', 'asr.emasses:main'}:
        remove_keys.add('gpwfilename')
    unused_old_params = unused_old_params - remove_keys

    assert not unused_old_params, (
        f'record.name={name}: Unused parameters from old record={unused_old_params}.'
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
