"""Module for registering dependencies between recipes."""


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
        print('dep getattr', attr)
        obj_attr = getattr(self.dependant_obj, attr)
        if attr.startswith('__'):
            return obj_attr
        return Dependant(
            obj=obj_attr,
            dependencies=self.dependencies,
        )

    def __setattr__(self, name, value):
        if name in ['dependant_obj', 'dependencies']:
            object.__setattr__(self, name, value)
        else:
            setattr(self.dependant_obj, name, value)

    def __getitem__(self, item):
        return Dependant(self.dependant_obj[item],
                         dependencies=self.dependencies)


methods = [
    '__add__',
    '__sub__',
    '__mul__',
    '__matmul__',
    '__truediv__',
    '__floordiv__',
    '__mod__',
    '__divmod__',
    '__pow__',
    '__lshift__',
    '__rshift__',
    '__and__',
    '__xor__',
    '__or__',
    '__radd__',
    '__rsub__',
    '__rmul__',
    '__rmatmul__',
    '__rtruediv__',
    '__rfloordiv__',
    '__rmod__',
    '__rdivmod__',
    '__rpow__',
    '__rlshift__',
    '__rrshift__',
    '__rand__',
    '__rxor__',
    '__ror__',
    '__iadd__',
    '__isub__',
    '__imul__',
    '__imatmul__',
    '__itruediv__',
    '__ifloordiv__',
    '__imod__',
    '__ipow__',
    '__ilshift__',
    '__irshift__',
    '__iand__',
    '__ixor__',
    '__ior__',
    '__neg__',
    '__pos__',
    '__abs__',
    '__invert__',
    '__complex__',
    '__int__',
    '__float__',
    '__index__',
    '__round__',
    '__trunc__',
    '__floor__',
    '__ceil__',
]


def make_method(method_name):

    def method(self, *args, **kwargs):
        return getattr(self.dependant_obj, method_name)(*args, **kwargs)

    return method


for meth in methods:

    method = make_method(meth)
    setattr(Dependant, meth, method)


def find_dependencies(dct):
    dependencies = []
    for key, value in dct.items():
        if isinstance(value, Dependant):
            dependencies.extend(value.dependencies)
        elif isinstance(value, dict):
            dependencies.extend(find_dependencies(value))
    return dependencies
