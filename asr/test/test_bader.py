"""Test bader recipe."""
import subprocess
import pytest
from asr.c2db.bader import main, Result, webpanel
from .materials import Si


@pytest.mark.ci
def test_bader(mockgpaw, asr_tmpdir, monkeypatch):
    monkeypatch.setattr(subprocess, 'run',
                        lambda args, stdout=None, stderr=None: None)
    result = main(atoms=Si, grid_spacing=0.05)
    assert (result.bader_charges == [-0.5, 0.5]).all()
    assert result.sym_a == ['Si', 'Si']


class DummyRow:
    pass


@pytest.mark.xfail
# row.data cannot be a dict since we want to call data.get_record() somewhere.
def test_bader_webpanel():
    result = Result(data=dict(bader_charges=[-0.5, 0.5], sym_a=['O', 'H']))
    row = DummyRow()
    row.data = {'results-asr.bader.json': result}
    panel, = webpanel(result, row, {})
    assert panel['columns'][0][0]['rows'][0] == ['0', 'O', '-0.50']
