import uuid
import fnmatch
import pathlib
from .params import Parameters
from .specification import RunSpecification


def is_migratable(obj):
    if hasattr(obj, 'migrate'):
        if obj.migrate is not None:
            return True
    return False


class NoMigrationError(Exception):

    pass


class Migrations:

    def __init__(self, generators, cache):

        self.generators = generators
        self.cache = cache
        self.done = {record.migration_id
                     for record in cache.select(include_migrated=True)
                     if record.migration_id}

    def __bool__(self):
        try:
            next(self.generate_migrations())
            return True
        except StopIteration:
            return False

    def generate_migrations(self):
        for generator in self.generators:
            migrations = generator(self.cache)
            for migration in migrations:
                if migration.id not in self.done:
                    yield migration

    def apply(self):
        from asr.core.specification import get_new_uuid
        for migration in self.generate_migrations():
            print(migration)
            original_record, migrated_record = migration.apply()
            self.done.add(migration.id)
            migrated_uid = get_new_uuid()
            migrated_record.run_specification.uid = migrated_uid
            migrated_record.migration_id = migration.id

            if original_record:
                original_uid = original_record.uid
                migrated_record.migrated_from = original_uid
                original_record.migrated_to = migrated_uid
                self.cache.update(original_record)
            self.cache.add(migrated_record)

    def __str__(self):
        lines = []
        for migration in self.generate_migrations():
            lines.append(str(migration))
        return '\n'.join(lines)


class Migration:

    def __init__(self, func, name=None, args=None, kwargs=None):

        self.func = func
        if args is None:
            args = ()
        self.args = args
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        if name is None:
            name = func.__name__ + f' args={self.args} kwargs={self.kwargs}'
        self.name = name
        self.id = name

    def apply(self):
        return self.func(*self.args, **self.kwargs)

    def __str__(self):
        return self.name


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
    from .record import RunRecord
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
    result = recipe.returns(
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

    return RunRecord(
        run_specification=RunSpecification(
            name=name,
            parameters=parameters,
            version=-1,
            codes=code_versions,
            uid=uuid.uuid4().hex,
        ),
        resources=resources,
        result=result,
        tags=['C2DB'],
    )


def get_old_records():
    records = []
    for resultsfile in find_results_files():
        record = construct_record_from_resultsfile(resultsfile)
        records.append(record)
    return records


def add_resultsfile_record(cache, resultsfile):
    record = construct_record_from_resultsfile(resultsfile)
    return None, record


def generate_resultsfile_migrations(cache):
    for resultsfile in find_results_files():
        yield Migration(
            add_resultsfile_record,
            name='Migrate resultsfile ' + str(resultsfile),
            args=(cache, resultsfile))


def generate_record_migrations(cache):

    migrations = []
    for record in cache.select():
        record_migrations = record.get_migrations(cache)
        if record_migrations:
            migrations.extend(record_migrations)

    for migration in migrations:
        yield migration
