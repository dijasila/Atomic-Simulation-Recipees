"""Module for registering dependencies between recipes."""
from .params import Parameters


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


def find_dependencies(obj):  # noqa
    dependencies = []
    dependencies.extend(get_dependencies(obj))

    values = get_values_of_object(obj)
    for value in values:
        dependencies.extend(find_dependencies(value))
    return dependencies


DEPATTR = '__deps__'


def get_values_of_object(obj):  # noqa
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


def mark_dependencies(obj, uid):  # noqa
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
            uids = find_dependencies(value)
            self.register_uids(uids)

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
                mark_dependencies(run_record.result, run_record.uid)
                for uid in dependencies:
                    mark_dependencies(run_record.result, uid)
                run_record.dependencies = dependencies
                return run_record

            return wrapped
        return wrapper

    def register_uids(self, uids):  # noqa
        dependencies = self.dependency_stack[-1]
        for uid in uids:
            if uid not in dependencies:
                dependencies.append(uid)

    def register(self, func):
        """Register dependency."""
        def wrapped(*args, **kwargs):
            run_record = func(*args, **kwargs)

            if self.dependency_stack:
                self.register_uids([run_record.uid])

            return run_record
        return wrapped


register_dependencies = RegisterDependencies()
