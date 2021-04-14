from .comparators import comparators
from .decorators import option
from .types import AtomsFile, DictStr


def atomsopt(function=None, *,
             default='structure.json',
             help='Atomic structure.'):
    """Make atoms CLI option.

    CLI option will have aliases "-a" and "--atoms".
    """
    opt = option('-a', '--atoms', help=help,
                 type=AtomsFile(), default=default)
    if function is None:
        return opt
    return opt(function)


def calcopt(function=None, *,
            aliases=None,
            help='Calculator params.'):
    """Make calculator CLI option.

    Default aliases is '-c', '--calculator'.
    """
    if aliases is None:
        aliases = ['-c', '--calculator']
    opt = option(*aliases,
                 help=help, type=DictStr(),
                 matcher=comparators.CALCULATORSPEC)
    if function is None:
        return opt
    return opt(function)
