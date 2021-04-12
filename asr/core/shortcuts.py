from .comparators import comparators
from .command import option
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


calcopt = option('-c', '--calculator',
                 help='Calculator params.', type=DictStr(),
                 matcher=comparators.CALCULATORSPEC)
