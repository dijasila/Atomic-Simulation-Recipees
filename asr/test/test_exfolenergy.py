import pytest
from asr.exfoliationenergy import get_bilayers_energies
from asr.exfoliationenergy import vdw_energy, calculate_exfoliation
from asr.core import write_json
from pathlib import Path
import os
from ase import Atoms
import numpy as np


def dont_test_get_bilayers_energies(asr_tmpdir):
    expected = [("f1", 1.0),
                ("f2", 2.5),
                ("f3", 3.3)]

    def make_expected(n, v):
        if not os.path.isdir(f"{asr_tmpdir}/{n}"):
            os.mkdir(f"{asr_tmpdir}/{n}")
        fn = f"{asr_tmpdir}/{n}/results-asr.relax_bilayer.json"
        d = {"energy": v}
        write_json(fn, d)

    for n, v in expected:
        make_expected(n, v)

    p = Path(asr_tmpdir)

    actual = get_bilayers_energies(p)

    for n, v in actual:
        fn = n.split("/")[-1]
        assert (fn, v) in expected


@pytest.mark.ci
def test_v_energy(asr_tmpdir):
    bilayers_energies = [("A", 1.0),
                         ("B", 2.0),
                         ("C", 3.0),
                         ("S", -1.0)]
    ml_e = np.random.rand() * 3
    vdw_e = np.random.rand()

    atoms = Atoms("H", cell=(5, 5, 5))
    area = 5 * 5
    exf_e, most_stable, bilayers = calculate_exfoliation(ml_e,
                                                         vdw_e,
                                                         bilayers_energies,
                                                         atoms)

    assert np.allclose(exf_e * area, 2 * (ml_e + vdw_e) - (-1.0)), exf_e
    assert most_stable == "S"
    assert all(x in bilayers for x in ["A", "B", "C", "S"])
