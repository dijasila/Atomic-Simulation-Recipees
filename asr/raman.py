"""Raman spectrum."""
from asr.core import command
from asr.result.resultdata import RamanResult


@command('asr.raman', returns=RamanResult)
def main() -> RamanResult:
    raise NotImplementedError
