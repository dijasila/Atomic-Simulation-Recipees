"""Test bader recipe."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from asr.bader import Result, bader, webpanel

bader_out = """
...
  NUMBER OF BADER MAXIMA FOUND:             18
      SIGNIFICANT MAXIMA FOUND:              2
           NUMBER OF ELECTRONS:        1.99999
...
"""

ACF_dat = """\
    #         X           Y           Z        CHARGE     MIN DIST    ATOMIC VOL
 -------------------------------------------------------------------------------
    1      4.7243      4.7243      4.7243      1.0000      0.6992    497.6226
    2      6.1227      4.7243      4.7243      1.0000      0.6088    470.7582
 -------------------------------------------------------------------------------
   NUMBER OF ELECTRONS:        1.99999
"""


class FakeGPAW:
    """Fake GPAW implementation."""

    def __init__(self, atoms):
        self.atoms = atoms

    def get_all_electron_density(self, gridrefinement):
        return np.ones((2, 3, 4))


def run(args, stdout, stderr):
    """Fake subprocess.run() function."""
    assert args == ['bader', 'density.cube']
    stdout.write(bader_out)
    Path('ACF.dat').write_text(ACF_dat)


@pytest.mark.ci
def test_bader(asr_tmpdir, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', run)
    gs = FakeGPAW(Atoms('H2'))
    atoms, charges = bader(gs)
    assert (charges == [0.0, 0.0]).all()


class DummyRow:
    pass


def test_bader_webpanel():
    result = Result(data=dict(bader_charges=[-0.5, 0.5], sym_a=['O', 'H']))
    row = DummyRow()
    row.data = {'results-asr.bader.json': result}
    panel, = webpanel(result, row, {})
    assert panel['columns'][0][0]['rows'][0] == ['0', 'O', '-0.50']
