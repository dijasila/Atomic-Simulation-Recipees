"""Test bader recipe."""
import subprocess
import pytest
from asr.bader import main
from .materials import Si


@pytest.mark.ci
def test_bader(mockgpaw, asrt_tmpdir, monkeypatch):
    monkeypatch.setattr(subprocess, 'run',
                        lambda args, stdout=None, stderr=None: None)
    Si.write('structure.json')
    result = main(0.05)
    assert (result.bader_charges == [0.5, -0.5]).all()
    assert result.sym_a == ['Si', 'Si']
