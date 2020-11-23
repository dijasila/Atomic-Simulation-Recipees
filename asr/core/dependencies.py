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


def find_dependencies(dct):
    dependencies = []
    for key, value in dct.items():
        if isinstance(value, Dependant):
            dependencies.extend(value.dependencies)
        elif isinstance(value, dict):
            dependencies.extend(find_dependencies(value))
    return dependencies
