import uuid
import fnmatch
import pathlib
from .parameters import Parameters
from .specification import RunSpecification
from .record import Record


def is_migratable(obj):
    if hasattr(obj, 'migrate'):
        if obj.migrate is not None:
            return True
    return False


class NoMigrationError(Exception):

    pass


class Migrations:

    def __init__(self, generator, cache):
        self.generator = generator
        self.cache = cache

    def __bool__(self):
        try:
            next(self.generate_migrations())
            return True
        except StopIteration:
            return False

    def generate_migrations(self):
        migrations = self.generator(self.cache)
        for migration in migrations:
            yield migration

    def apply(self):
        from asr.core.specification import get_new_uuid
        for migration in self.generate_migrations():
            print('printing migration')
            print(migration)
            records = migration.apply()

            for original_record, migrated_record in zip(
                    records[:-1], records[1:]):
                migrated_uid = get_new_uuid()
                migrated_record.run_specification.uid = migrated_uid

                original_uid = original_record.uid
                migrated_record.migrated_from = original_uid
                original_record.migrated_to = migrated_uid

            original_record, *migrated_records = records
            self.cache.update(original_record)
            for migrated_record in migrated_records:
                self.cache.add(migrated_record)

    def __str__(self):
        lines = []
        for migration in self.generate_migrations():
            lines.append(str(migration))
        return '\n'.join(lines)


class Migration:

    def __init__(
            self, func, from_version, to_version, name=None,
            dep: 'Migration' = None, record=None):

        self.func = func
        if name is None:
            name = func.__name__
        self.name = name
        if dep:
            assert not record
        self.dep = dep
        self.record = record
        self.from_version = from_version
        self.to_version = to_version

    def apply(self) -> Record:
        if self.dep:
            migrated_records = self.dep.apply()
            migrated_record = self.func(migrated_records[-1].copy())
            migrated_record.version = self.to_version
            return [*migrated_records, migrated_record]
        else:
            migrated_record = self.func(self.record.copy())
            migrated_record.version = self.to_version
            return [self.record, migrated_record]

    def __str__(self):
        text = f'{self.name} {self.from_version} -> {self.to_version}'
        if self.dep:
            return ' '.join([str(self.dep), text])
        return text


def find_results_files():
    relevant_patterns = [
        'results-asr.*.json',
        'displacements*/*/results-*.json',
        'strains*/results-*.json',
    ]

    skip_patterns = [
        'results-asr.database.fromtree.json',
        'results-asr.database.app.json',
        'results-asr.database.key_descriptions.json',
        'displacements*/*/results-asr.database.material_fingerprint.json',
        'strains*/results-asr.database.material_fingerprint.json',
        '*asr.setinfo*',
        '*asr.setup.params.json',
        '*asr.setup.params.json',
    ]

    for pattern in relevant_patterns:
        for path in pathlib.Path().glob(pattern):
            filepath = str(path)
            if any(fnmatch.fnmatch(filepath, skip_pattern)
                   for skip_pattern in skip_patterns):
                continue
            yield path


def construct_record_from_resultsfile(path):
    from .record import Record
    from asr.core.results import MetaDataNotSetError
    from asr.core import read_json, get_recipe_from_name, ASRResult
    from ase.io import read
    path = pathlib.Path(path)
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
                paths=['gs.gpw'],
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

    atoms = read(path.parent / 'structure.json')
    try:
        parameters = result.metadata.params
    except MetaDataNotSetError:
        parameters = {}

    parameters = Parameters(parameters)
    if 'atoms' not in parameters:
        parameters.atoms = atoms.copy()

    try:
        code_versions = result.metadata.code_versions
    except MetaDataNotSetError:
        code_versions = {}

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

    return Record(
        run_specification=RunSpecification(
            name=name,
            parameters=parameters,
            version=-1,
            codes=code_versions,
            uid=uuid.uuid4().hex,
        ),
        resources=resources,
        result=result,
    )


def get_old_records():
    records = []
    for resultsfile in find_results_files():
        record = construct_record_from_resultsfile(resultsfile)
        records.append(record)
    return records


def add_resultsfile_record(resultsfile):
    record = construct_record_from_resultsfile(resultsfile)
    return None, record


def get_resultsfile_records():
    records = []
    for resultsfile in find_results_files():
        record = construct_record_from_resultsfile(resultsfile)
        records.append(record)
    return records


def generate_record_migrations(cache):
    for record in cache.select():
        record_migration = record.get_migration()
        if record_migration:
            yield record_migration
