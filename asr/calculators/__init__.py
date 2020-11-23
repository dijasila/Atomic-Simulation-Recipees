from ase import Atoms
import copy
import typing


def default(atoms, dct):
    return dct


def asr_gpaw_parameter_factory(atoms, dct):
    dct = copy.deepcopy(dct)
    nd = sum(atoms.pbc)
    if nd == 2:
        assert not atoms.get_pbc()[2], \
            ('The third unit cell axis should be aperiodic for '
             'a 2D material!')
        dct['poissonsolver'] = {'dipolelayer': 'xy'}

    precision = dct.pop('precision', None)
    assert precision in {'low', 'high', None}
    if precision == 'low':
        dct.update({'mode': {'name': 'pw', 'ecut': 350},
                    'kpts': {'density': 2.0, 'gamma': True},
                    'symmetry': {'symmorphic': False},
                    'convergence': {'forces': 1e-3}})
    elif precision == 'high':
        dct.update({'mode': {'name': 'pw', 'ecut': 800},
                    'kpts': {'density': 12.0, 'gamma': True},
                    'symmetry': {'symmorphic': False},
                    'convergence': {'forces': 1e-4},
                    'charge': 0})
    return dct


asr_parameter_factories = {'gpaw': asr_gpaw_parameter_factory,
                           'default': default}


def get_parameter_factory(name):
    factory = asr_parameter_factories.get(
        name,
        asr_parameter_factories['default'],
    )
    return factory


def get_calculator_spec(atoms: Atoms, dct: typing.Dict[str, typing.Any]):
    calculatorname = dct['name']
    factory = get_parameter_factory(calculatorname)
    calcspec = factory(atoms, dct)
    return calcspec


def set_calculator_hook(parameters):
    from asr.calculators import get_calculator_spec
    atoms = parameters.atoms
    calc_spec = get_calculator_spec(atoms, parameters.calculator)
    parameters.calculator = calc_spec
    return parameters
