import typing

from asr.calculators import set_calculator_hook

from .command import get_recipes
from .old_defaults import get_old_defaults
from .utils import get_recipe_from_name
from .migrate import mutation
from .selector import Selector
from .parameters import Parameters


def param_is_gpaw_calculator(param: typing.Any):
    if not isinstance(param, dict):
        return False
    if 'mode' in param:
        return True
    elif 'nbands' in param:
        return True
    return False


def param_is_missing_gpaw_calculator(param: typing.Any):
    if param_is_gpaw_calculator(param):
        if 'name' not in param:
            return True
    return False


def select_records_that_misses_names_in_gpaw_calculator(record):
    if record.version < 0:
        return False
    for _, value in record.parameters.items():
        if param_is_missing_gpaw_calculator(value):
            return True
    return False


@mutation(
    selector=select_records_that_misses_names_in_gpaw_calculator,
    eagerness=5,
)
def add_missing_names_in_gpaw_calculators(record):
    """Add missing names in GPAW calculators."""
    for name, value in record.parameters.items():
        if param_is_missing_gpaw_calculator(value):
            value['name'] = 'gpaw'
    return record


def select_records_that_have_kpts_density_specified(record):
    if "atoms" not in record.parameters:
        return False
    # In some cases the input atoms were unknown
    # and resultfile.py assigns "Unknown atoms file" to these
    if record.parameters.atoms == "Unknown atoms file":
        return False
    for name, value in record.parameters.items():
        if 'calculator' in name and 'kpts' in value and 'density' in value['kpts']:
            return True
    return False


@mutation(selector=select_records_that_have_kpts_density_specified, eagerness=-2)
def apply_calculator_hook_to_old_records(record):
    """Fix abstract calculator values to more concrete values."""
    parameters = set_calculator_hook(record.parameters)
    record.parameters = parameters
    return record


sel = Selector()
sel.version = sel.EQ(-1)


@mutation(selector=sel, uid="9269242a035a4731bcd5ac609ff0a086", eagerness=-1)
def update_resultfile_record_to_version_0(record):
    """Extract missing parameters from dependencies, add them and set version to 0."""
    default_params = get_old_defaults()[record.name]
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
                    f'Please add a mutation for {name} that fixes these issues '
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
    # Write defaults to a file.
    from .serialize import JSONSerializer
    from .utils import write_file
    import pathlib
    PATH = pathlib.Path(__file__).parent / "old_resultfile_defaults.json"
    defaults = get_defaults_from_all_recipes()
    jsontxt = JSONSerializer().serialize(defaults)
    write_file(PATH, jsontxt)
