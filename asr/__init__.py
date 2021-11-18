"""Top-level package for Atomic Simulation Recipes."""
from asr.core import (  # noqa
    command,
    option,
    argument,
    migration,
    Selector,
    comparators,
    atomsopt,
    calcopt,
    instruction,
    Record,
    RunSpecification,
    Resources,
    Dependencies,
    RevisionHistory,
    Metadata,
    NonMigratableRecord,
)

matchers = comparators
__author__ = """Morten Niklas Gjerding"""
__email__ = 'mortengjerding@gmail.com'
__version__ = '0.4.1'
name = "asr"

__all__ = [
    'command',
    'option',
    'argument',
    'migration',
    'Selector',
    'matchers',
    'atomsopt',
    'calcopt',
    'instruction',
]
