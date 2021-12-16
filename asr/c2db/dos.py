"""Density of states."""
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.dft.dos import DOS

import asr
from asr.c2db.gs import calculate as gscalculate


def webpanel(result, context):
    from asr.database.browser import fig

    panel = {
        "title": f"Density of states ({context.xcname})",
        "columns": [[fig("dos.png")], []],
        "plot_descriptions": [
            {
                "function": plot,
                "filenames": ["dos.png"],
            },
        ],
        "sort": 10,
    }
    return [panel]


@asr.prepare_result
class Result(asr.ASRResult):

    energies_e: List[float]
    dosspin0_e: List[float]
    dosspin1_e: List[float]
    natoms: int
    volume: float

    key_descriptions: Dict[str, str] = dict(
        energies_e="Energy grid sampling [eV]",
        dosspin0_e="DOS for spin channel 0 [#/unit cell]",
        dosspin1_e="DOS for spin channel 1 [#/unit cell]",
        natoms="Number of atoms",
        volume="Volume of unit cell [Å^3]",
    )

    formats = {"webpanel2": webpanel}


@asr.instruction("asr.c2db.dos")
@asr.atomsopt
@asr.calcopt
@asr.option("--kptdensity", help="K point kptdensity", type=float)
def main(
    atoms: Atoms,
    calculator: Dict = gscalculate.defaults.calculator,
    kptdensity: float = 12.0,
) -> Result:
    """Calculate DOS."""
    from gpaw import GPAW  # noqa

    result = gscalculate(atoms=atoms, calculator=calculator)
    name = "dos.gpw"
    if not Path(name).is_file():
        calc = result.calculation.load(
            kpts=dict(density=kptdensity),
            nbands="300%",
            convergence={"bands": -10},
        )
        calc.get_potential_energy()
        calc.write(name)
        del calc

    calc = GPAW(name, txt=None)

    dos = DOS(calc, width=0.0, window=(-5, 5), npts=1000)
    nspins = calc.get_number_of_spins()
    dosspin0_e = dos.get_dos(spin=0)
    energies_e = dos.get_energies()
    natoms = len(calc.atoms)
    volume = calc.atoms.get_volume()
    data = {
        "dosspin0_e": dosspin0_e.tolist(),
        "energies_e": energies_e.tolist(),
        "natoms": natoms,
        "volume": volume,
    }
    if nspins == 2:
        dosspin1_e = dos.get_dos(spin=1)
        data["dosspin1_e"] = dosspin1_e.tolist()

    return Result.fromdata(**data)


def plot(context, filename):
    """Plot DOS.

    Defaults to dos.json.
    """
    dos = context.result

    plt.figure()
    plt.plot(dos["energies_e"], np.array(dos["dosspin0_e"]) / dos["volume"])
    plt.xlabel(r"Energy - $E_\mathrm{F}$ (eV)")
    plt.ylabel(r"DOS (states / (eV Å$^3$)")
    plt.tight_layout()
    plt.savefig(filename)


sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.name = sel.EQ("asr.c2db.dos:main")
sel.parameters = sel.CONTAINS("name")


@asr.mutation(selector=sel)
def remove_name_from_params(record: asr.Record) -> asr.Record:
    """Remove name param from record."""
    del record.parameters.name
    return record


sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.name = sel.EQ("asr.c2db.dos:main")
sel.parameters = sel.CONTAINS("filename")


@asr.mutation(selector=sel)
def remove_filename_from_params(record: asr.Record) -> asr.Record:
    """Remove filename param from record."""
    del record.parameters.filename
    return record


sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.name = sel.EQ("asr.c2db.dos:main")
sel.parameters = sel.NOT(sel.CONTAINS("calculator"))


@asr.mutation(selector=sel)
def add_calculator_to_params(record: asr.Record) -> asr.Record:
    """Add calculator to parameters."""
    record.parameters.calculator = {
        "name": "gpaw",
        "mode": {"name": "pw", "ecut": 800},
        "xc": "PBE",
        "kpts": {"density": 12.0, "gamma": True},
        "occupations": {"name": "fermi-dirac", "width": 0.05},
        "convergence": {"bands": "CBM+3.0"},
        "nbands": "200%",
        "txt": "gs.txt",
        "charge": 0,
    }
    return record


if __name__ == "__main__":
    main.cli()
