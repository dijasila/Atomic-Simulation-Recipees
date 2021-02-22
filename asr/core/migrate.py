"""Implements record migration functionality."""
import fnmatch
import pathlib
import typing
from dataclasses import dataclass
from .selector import Selector
from .parameters import Parameters
from .specification import RunSpecification, get_new_uuid
from .record import Record


class NoMigrationError(Exception):
    """Raise when no migration are needed."""


RecordUID = str


@dataclass
class MigrationLog:
    """Container for logging migration information."""

    migrated_from: typing.Optional[RecordUID]
    migrated_to: typing.Optional[RecordUID]


@dataclass
class RecordMutation:
    """A class to update a record to a greater version."""

    function = typing.Callable
    from_version: int
    to_version: int
    selector: Selector
    description: str

    def apply(self, record: Record) -> Record:
        """Apply mutation to record and return mutated record."""
        assert self.applies_to(record)
        migrated_record = self.function(record.copy())
        migrated_record.version = self.to_version
        migration_log = MigrationLog(migrated_from=record.uid)
        migrated_record.migration_log = migration_log
        return migrated_record

    def applies_to(self, record: Record) -> bool:
        """Check if mutation applies to record."""
        return self.selector.matches(record)

    def __str__(self):
        return f'{self.description} {self.from_version} -> {self.to_version}'


class RecordMigrationFactory:
    """Construct record migrations.

    Manages a collection of RecordMutations and can be told to create
    RecordMigration.
    """

    def __init__(self, mutations):
        self.mutations = []
        for mutation in mutations:
            self.mutations.append(mutation)

    def add(self, mutation: RecordMutation):
        """Add mutation."""
        self.mutations.append(mutation)

    def __call__(self, record: Record) -> typing.Optional['RecordMigration']:
        try:
            sequence_of_mutations = make_migration_strategy(
                self.mutations, record)
        except NoMigrationError:
            return None
        record_migration = RecordMigration(sequence_of_mutations, record)
        return record_migration


def make_migration_strategy(
    mutations: typing.List[RecordMutation],
    record: Record,
) -> typing.List[RecordMutation]:
    relevant_mutations = {}
    for mutation in mutations:
        if mutation.applies_to(record):
            relevant_mutations[mutation.from_version] = mutation

    strategy = []
    version = record.version
    if version not in relevant_mutations:
        raise NoMigrationError
    while version in relevant_mutations:
        mutation = relevant_mutations[version]
        strategy.append(mutation)
        version = mutation.to_version
    return strategy


@dataclass
class RecordMigration:
    migrations: typing.List[RecordMutation]
    record: Record

    def apply(self, cache):
        records = [self.record]
        for migration in self.migrations:
            migrated_record = migration(records[-1])
            records.append(migrated_record)

        original_record, *migrated_records = records
        cache.update(original_record)
        for migrated_record in migrated_records:
            cache.add(migrated_record)


def find_directories() -> typing.List[pathlib.Path]:
    directories = []
    for path in pathlib.Path().rglob('*'):
        if path.is_dir():
            directories.append(path)

    return directories


def generate_uids(resultfiles) -> typing.Dict[pathlib.Path, str]:
    return {path: get_new_uuid() for path in resultfiles}


def find_results_files() -> typing.List[pathlib.Path]:
    # relevant_patterns = [
    #     'results-asr.*.json',
    #     'displacements*/*/results-*.json',
    #     'strains*/results-*.json',
    # ]

    skip_patterns = [
        'results-asr.database.fromtree.json',
        'results-asr.database.app.json',
        'results-asr.database.key_descriptions.json',
        'results-asr.setup.strains*.json',
        'displacements*/*/results-asr.database.material_fingerprint.json',
        'strains*/results-asr.database.material_fingerprint.json',
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

    # for pattern in relevant_patterns:
    #     for path in pathlib.Path().glob(pattern):
    #         filepath = str(path)
    #         if any(fnmatch.fnmatch(filepath, skip_pattern)
    #                for skip_pattern in skip_patterns):
    #             continue
    #         paths.append(path)
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
            metadata = {'asr_name': recipename}
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
        name += '::main'
    else:
        name = name.replace('@', '::')

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
        tags=['Generated from result file.'],
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


# def get_old_records():
#     records = []
#     for resultsfile in find_results_files():
#         record = construct_record_from_resultsfile(resultsfile)
#         records.append(record)
#     return records


# def add_resultsfile_record(resultsfile):
#     record = construct_record_from_resultsfile(resultsfile)
#     return None, record


def get_resultsfile_records() -> typing.List[Record]:
    resultsfiles = find_results_files()
    records = []
    uids = generate_uids(resultsfiles)
    for path in resultsfiles:
        record = construct_record_from_resultsfile(path, uids)
        records.append(record)

    return records


def generate_record_migrations(cache):
    for record in cache.select():
        record_migration = record.get_migration()
        if record_migration:
            yield record_migration
