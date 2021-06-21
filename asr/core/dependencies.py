"""Module for registering dependencies between recipes."""
import textwrap
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

    def __str__(self):
        return f'uid={self.uid} revision={self.revision}'


# XXX: We cannot simply subclass list since this breaks serialization.
# I don't have time to fix this atm.


@dataclasses.dataclass
class Dependencies:

    deps: list[Dependency] = dataclasses.field(default_factory=list)

    def __str__(self):
        lines = []
        for dependency in self:
            value = str(dependency)
            if '\n' in value:
                value = '\n' + textwrap.indent(value, ' ')
            lines.append(f'dependency={value}')
        return '\n'.join(lines)

    def __repr__(self):
        items = [item for item in self.deps]
        return f'Dependencies({items})'

    def extend(self, value: 'Dependencies'):
        self.deps.extend(value.deps)

    def append(self, value: Dependency):
        self.deps.append(value)

    def __getitem__(self, item):
        return self.deps[item]

    def __iter__(self):
        for value in self.deps:
            yield value

    def __contains__(self, value):
        return value in self.deps


def construct_dependency(record):
    return Dependency(record.uid, record.revision)


def find_dependencies(obj):
    dependencies = Dependencies()
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
            if not key == '__deps__':
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
    deps = get_dependencies(obj)
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
    """Get data dependencies."""
    return getattr(obj, DEPATTR, Dependencies())


dependency_stack = []


class RegisterDependencies:
    """Register dependencies."""

    def __init__(self, dependency_stack=dependency_stack):  # noqa
        self.dependency_stack = dependency_stack

    def __enter__(self):
        """Add frame to dependency stack."""
        dependencies = Dependencies()
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
