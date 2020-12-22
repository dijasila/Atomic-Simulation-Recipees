from .command import option
from .types import AtomsFile, DictStr

atomsopt = option('-a', '--atoms', help='Atomic structure.',
                  type=AtomsFile(), default='structure.json')

calcopt = option('-c', '--calculator',
                 help='Calculator params.', type=DictStr())
