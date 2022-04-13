"""Implement ASR calculator utilities.

Implement parameter factories and calculator adapters.

"""

import pathlib
import copy
import typing
from ase import Atoms
from ase.calculators.calculator import get_calculator_class \
    as ase_get_calculator_class
from ase.calculators.calculator import Calculator, kptdensity2monkhorstpack
from abc import ABC, abstractmethod


def default(atoms, dct):
    """Modify parameters (default doesn't do anything)."""
    return dct


def asr_gpaw_parameter_factory(atoms, dct):
    """Modify parameters according to dimensionality."""
    dct = copy.deepcopy(dct)
    nd = sum(atoms.pbc)
    if nd == 2:
        assert not atoms.get_pbc()[2], \
            ('The third unit cell axis should be aperiodic for '
             'a 2D material!')
        dct['poissonsolver'] = {'dipolelayer': 'xy'}

    precision = dct.pop('precision', None)
    assert precision in {'low', 'medium', 'high', None}
    if precision == 'low':
        dct.update({'mode': {'name': 'pw', 'ecut': 350},
                    'kpts': {'density': 2.0, 'gamma': True},
                    'symmetry': {'symmorphic': False},
                    'convergence': {'forces': 1e-3}})
    elif precision == 'medium':
        dct.update({'mode': {'name': 'pw', 'ecut': 500},
                    'kpts': {'density': 4.0, 'gamma': True},
                    'symmetry': {'symmorphic': False},
                    'convergence': {'forces': 1e-3}})
    elif precision == 'high':
        dct.update({'mode': {'name': 'pw', 'ecut': 800},
                    'kpts': {'density': 12.0, 'gamma': True},
                    'symmetry': {'symmorphic': False},
                    'convergence': {'forces': 1e-4},
                    'charge': 0})

    if 'kpts' in dct:
        kpts = dct['kpts']
        if 'density' in kpts:
            density = kpts.pop('density')
            even = kpts.pop('even', False)
            size = kptdensity2monkhorstpack(
                atoms=atoms,
                kptdensity=density,
                even=even,
            )
            kpts['size'] = size
    return dct


asr_parameter_factories = {'gpaw': asr_gpaw_parameter_factory,
                           'default': default}


def get_parameter_factory(name):  # noqa
    factory = asr_parameter_factories.get(
        name,
        asr_parameter_factories['default'],
    )
    return factory


def get_calculator_spec(atoms: Atoms, dct: typing.Dict[str, typing.Any]):
    """Create paramter spec by applying paramter factory."""
    calculatorname = dct['name']
    factory = get_parameter_factory(calculatorname)
    calcspec = factory(atoms, dct)
    return calcspec


def set_calculator_hook(parameters):
    """Set parameters according to dimensionality."""
    from asr.calculators import get_calculator_spec
    if 'atoms' not in parameters:
        return parameters
    atoms = parameters.atoms
    for name, value in parameters.items():
        if 'calculator' in name:
            calc_spec = get_calculator_spec(atoms, parameters[name])
            parameters[name] = calc_spec
    return parameters


class Calculation:
    """Persist calculation state."""

    def __init__(self, id, cls_name, state=None, *, paths):  # noqa
        self.id = id
        self.cls_name = cls_name
        from pathlib import Path
        self.paths = [Path(path) for path in paths]
        self.state = state

    def __repr__(self):
        return (f'<Calculation id={self.id}, '
                f'cls_name={self.cls_name}, '
                f'paths={self.paths}>')

    def load(self, *args, **kwargs) -> Calculator:
        """Restart calculation."""
        cls = get_calculator_class(self.cls_name)
        return cls.load(self, *args, **kwargs)


class ASRAdapter(ABC):

    def __init__(self, cls):
        self.cls = cls
        self.calculator = None

    def __call__(self, *args, **kwargs):
        self.calculator = self.cls(*args, **kwargs)
        return self

    def __getattr__(self, attr):  # noqa
        if hasattr(self.calculator, attr):
            return getattr(self.calculator, attr)
        raise AttributeError

    @abstractmethod
    def save(self, id) -> Calculation:
        pass

    @abstractmethod
    def load(cls, calculation: Calculation, *args, **kwargs) -> 'ASRAdapter':
        pass


class GPAWLikeAdapter(ASRAdapter):

    def save(self, id) -> Calculation:
        filename = f'{id}.gpw'
        self.write(filename)
        return Calculation(
            id=id,
            cls_name='gpaw',
            paths=[filename],
        )

    def load(self,
             calculation: Calculation,
             **kwargs) -> 'GPAWLikeAdapter':

        parallel = kwargs.pop('parallel', True)
        if parallel:
            self.calculator = self.cls(pathlib.Path(
                calculation.paths[0]), **kwargs)
            return self
        from gpaw.mpi import serial_comm
        self.calculator = self.cls(
            pathlib.Path(calculation.paths[0]),
            communicator=serial_comm,
            **kwargs,
        )
        return self


def get_calculator_class(name):
    """Get ASR-ASE calculator adapter.

    Fall back to ase get_calculator_class. Will not be compatible with
    the Calculation class in that case.

    """
    cls = ase_get_calculator_class(name)

    if name == 'gpaw':
        calc = GPAWLikeAdapter(cls)
        return calc

    elif name == 'quantumespresso':

        class QEAdapter(ASRAdapter):

            def save(self, id) -> Calculation:
                filenames = ['qe1.txt', 'qe2.txt']

                return Calculation(
                    id=id, cls_name='quantumespresso', paths=filenames,
                    state=self.__dict__,
                )

            @classmethod
            def load(cls, calculation: Calculation) -> 'QEAdapter':
                for side_effect in calculation.paths:
                    side_effect.restore()
                obj = cls.__new__(cls)
                obj.__dict__.update(calculation.state)
                return obj

        return QEAdapter

    elif name == 'emt':

        class EMTAdapter(cls):

            def save(self, id) -> Calculation:
                return Calculation(
                    id=id,
                    cls_name='emt',
                    state=self.calculator.__dict__,
                )

            @classmethod
            def load(cls, calculation: Calculation) -> 'EMTAdapter':
                obj = cls.__new__(cls)
                obj.__dict__.update(calculation.dct)
                return obj

        return EMTAdapter
    else:
        return cls


def construct_calculator(calcspec):
    name = calcspec.pop('name')
    return get_calculator_class(name)(**calcspec)
