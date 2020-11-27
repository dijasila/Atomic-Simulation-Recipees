"""Module for registering dependencies between recipes."""
import typing
import copy


class Dependant:

    def __init__(self, obj, dependencies: typing.List[str]):
        self.dependant_obj = obj
        self.dependencies = dependencies

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __copy__(self):
        return Dependant(self.dependant_obj, self.dependencies)

    def __deepcopy__(self, memo):
        return Dependant(
            copy.deepcopy(self.dependant_obj, memo),
            copy.deepcopy(self.dependencies)
        )

    def __call__(self, *args, **kwargs):
        return self.dependant_obj(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.dependant_obj, attr)


def find_dependencies(obj):
    dependencies = []
    dependencies.extend(get_dependencies(obj))

    values = get_values_of_object(obj)
    for value in values:
        dependencies.extend(find_dependencies(value))
    return dependencies


DEPATTR = '__deps__'


def get_values_of_object(obj):
    values = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            values.append(value)
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            values.append(value)
    elif hasattr(obj, '__slots__'):
        for key in obj.__slots__:
            value = getattr(obj, key)
            values.append(value)
    return values


def mark_dependencies(obj, uid):
    mark_dependency(obj, uid)

    values = get_values_of_object(obj)
    for value in values:
        mark_dependencies(value, uid)


def mark_dependency(obj, uid):
    """Mark that obj is data-dependent on data with "uid"."""
    deps = getattr(obj, DEPATTR, [])
    if uid not in deps:
        deps.append(uid)
        try:
            setattr(obj, DEPATTR, deps)
        except AttributeError:
            pass


def has_dependency(obj):
    """Check if object has data dependency."""
    if hasattr(obj, DEPATTR):
        return True
    return False


def get_dependencies(obj):
    """Get data dependencies.

    Return None if no deps.
    """
    return getattr(obj, DEPATTR, [])
