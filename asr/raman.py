"""Raman spectrum."""
from asr.core import command
from asr.paneldata import RamanResult


@command('asr.raman', returns=RamanResult)
def main() -> RamanResult:
    raise NotImplementedError
