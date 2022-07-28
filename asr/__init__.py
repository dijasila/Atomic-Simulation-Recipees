"""Top-level package for Atomic Simulation Recipes."""
from htwutil import *  # noqa
# TODO: figure out which specific toplevel things we need from htwutil

from asr.core import (  # noqa
    ASRResult,
    Dependencies,
    Metadata,
    Mutation,
    NonMigratableRecord,
    Parameters,
    Record,
    Resources,
    Revision,
    RevisionHistory,
    RunSpecification,
    Selector,
    argument,
    atomsopt,
    calcopt,
    command,
    comparators,
    get_cache,
    instruction,
    mutation,
    option,
    prepare_result,
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
    'mutation',
    'Selector',
    'matchers',
    'atomsopt',
    'calcopt',
    'instruction',
]
