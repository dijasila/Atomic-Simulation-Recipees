import typing
from .comparators import comparators


class NoSuchAttribute(Exception):

    pass


class SelectorSetter:

    def __init__(self, selection: 'Selector', attr):
        self.__dict__['selection'] = selection
        self.__dict__['attrs'] = [attr]

    def __getattr__(self, attr):
        self.attrs.append(attr)
        return self

    def __setattr__(self, attr, value):
        setattr(
            self.selection,
            '.'.join(self.attrs + [attr]),
            value,
        )

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, item, value):
        return self.__setattr__(item, value)


class Selector:

    # Shortcuts for comparison functions
    EQ = EQUAL = staticmethod(comparators.EQUAL)
    IS = staticmethod(comparators.IS)
    LT = LESS_THAN = staticmethod(comparators.LESS_THAN)
    GT = GREATER_THAN = staticmethod(comparators.GREATER_THAN)
    LTE = LESS_THAN_EQUALS = staticmethod(comparators.LESS_THAN_EQUALS)
    GTE = GREATER_THAN_EQUALS = staticmethod(comparators.GREATER_THAN_EQUALS)
    APPROX = staticmethod(comparators.APPROX)
    ATOMS_EQUAL_TO = staticmethod(comparators.ATOMS_EQUAL_TO)
    ANY = staticmethod(comparators.ANY)
    CONTAINS = staticmethod(comparators.CONTAINS)
    ALL = staticmethod(comparators.ALL)
    OR = staticmethod(comparators.OR)
    AND = staticmethod(comparators.AND)
    NOT = staticmethod(comparators.NOT)

    def __init__(self, **selection):
        self.__dict__['selection'] = {}

        for key, value in selection.items():
            setattr(self, key, value)

    def matches(self, obj) -> bool:

        for attr, comparator in self.selection.items():
            try:
                objvalue = get_attribute(obj, attr.split('.'))
            except NoSuchAttribute:
                return False
            if not comparator(objvalue):
                return False
        return True

    def __setattr__(self, attr, value):
        self.selection[attr] = value

    def __getattr__(self, attr):
        return SelectorSetter(self, attr)

    def __setitem__(self, item, value):
        return self.__setattr__(item, value)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __repr__(self):
        parts = [
            f'{key}={repr(value)}'
            for key, value in
            self.selection.items()
        ]
        return 'Selector(' + ', '.join(parts) + ')'

    def __call__(self, obj: typing.Any) -> bool:
        return self.matches(obj)


def get_attribute(obj, attrs):

    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            try:
                obj = obj[attr]
            except (TypeError, KeyError, AttributeError):
                raise NoSuchAttribute

    return obj
