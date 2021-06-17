"""Module for registering dependencies between recipes."""
import typing
from .parameters import Parameters
import dataclasses

UID = str

# class Dependant:

#     def __init__(self, obj, dependencies: typing.List[str]):
#         self.dependant_obj = obj
#         self.dependencies = dependencies

#     def __getstate__(self):
#         return self.__dict__

#     def __setstate__(self, state):
#         self.__dict__.update(state)

#     def __copy__(self):
#         return Dependant(self.dependant_obj, self.dependencies)

#     def __deepcopy__(self, memo):
#         return Dependant(
#             copy.deepcopy(self.dependant_obj, memo),
#             copy.deepcopy(self.dependencies)
#         )

#     def __call__(self, *args, **kwargs):
#         return self.dependant_obj(*args, **kwargs)

#     def __getattr__(self, attr):
#         return getattr(self.dependant_obj, attr)


@dataclasses.dataclass
class Dependency:

    uid: UID
    revision: UID


def construct_dependency(record):
    return Dependency(record.uid, record.revision)


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


def mark_dependencies(obj: typing.Any, dependency: Dependency):  # noqa
    mark_dependency(obj, dependency)

    values = get_values_of_object(obj)
    for value in values:
        mark_dependencies(value, dependency)


def mark_dependency(obj, dependency: Dependency):
    """Mark that obj is dependent on 'dependency'."""
    deps = getattr(obj, DEPATTR, [])
    if dependency not in deps:
        deps.append(dependency)
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

    Return [] if no deps.
    """
    return getattr(obj, DEPATTR, [])


dependency_stack = []


class RegisterDependencies:
    """Register dependencies."""

    def __init__(self, dependency_stack=dependency_stack):  # noqa
        self.dependency_stack = dependency_stack

    def __enter__(self):
        """Add frame to dependency stack."""
        dependencies = []
        self.dependency_stack.append(dependencies)
        return dependencies

    def parse_argument_dependencies(self, parameters: Parameters):  # noqa

        for key, value in parameters.items():
            dependencies = find_dependencies(value)
            for dependency in dependencies:
                self.register_dependency(dependency)

        return parameters

    def __exit__(self, type, value, traceback):
        """Pop frame of dependency stack."""
        self.dependency_stack.pop()

    def __call__(self):  # noqa

        def wrapper(func):

            def wrapped(run_specification):
                with self as dependencies:
                    parameters = self.parse_argument_dependencies(
                        run_specification.parameters
                    )
                    run_specification.parameters = parameters
                    run_record = func(run_specification)
                dependency = construct_dependency(run_record)
                mark_dependencies(run_record.result, dependency)
                for dependency in dependencies:
                    mark_dependencies(run_record.result, dependency)
                run_record.dependencies = dependencies
                return run_record

            return wrapped
        return wrapper

    def register_dependency(self, dependency: Dependency):
        dependencies = self.dependency_stack[-1]
        if dependency not in dependencies:
            dependencies.append(dependency)

    def register(self, func):
        """Register dependency."""
        def wrapped(*args, **kwargs):
            run_record = func(*args, **kwargs)

            dependency = construct_dependency(run_record)
            if self.dependency_stack:
                self.register_dependency(dependency)

            return run_record
        return wrapped


register_dependencies = RegisterDependencies()
