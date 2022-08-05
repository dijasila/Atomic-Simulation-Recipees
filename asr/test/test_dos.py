import numpy as np
import pytest
from ase import Atoms
from asr.dos import _main, dos_plot, main


class DOSCalculator:
    nspins = 2

    def raw_dos(self, energies, spin, width):
        return np.ones_like(energies)


def test_dos():
    data = _main(DOSCalculator())
    print(data)


class Row:
    def __init__(self, result):
        self.data = self
        self.result = result

    def get(self, name):
        assert name == 'results-asr.dos.json'
        return self.result


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_gpaw_dos_h(tmp_path):
    from gpaw import GPAW
    h = Atoms('H',
              cell=[2.0, 2.0, 1.0],
              pbc=[0, 0, 1])
    h.center(axis=(0, 1))
    h.calc = GPAW(mode={'name': 'pw',
                        'ecut': 300},
                  kpts=[1, 1, 10],
                  txt=None)
    h.get_potential_energy()
    gpw = tmp_path / 'gs.gpw'
    h.calc.write(gpw)
    dos = main.get_wrapped_function()
    dos(gpw)
    result = dos(gpw.with_name('dos.gpw'))
    dos_plot(Row(result), gpw.with_name('dos.png'))
