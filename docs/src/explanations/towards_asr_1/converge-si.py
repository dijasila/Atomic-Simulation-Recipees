"""Convergence study of silicon band gap."""

from asr.gs import main as gs
from ase.build import bulk

atoms = bulk('Si')

densities = [2, 3, 4]
cutoffs = [200, 300, 400]
records = []
for density in densities:
    for ecut in cutoffs:
        record = gs(
            atoms=atoms,
            calculator=dict(
                name='gpaw',
                kpts=dict(density=density),
                mode=dict(name='pw', ecut=ecut),
                txt=None,
            )
        )
        records.append(record)


from matplotlib import pyplot as plt
import numpy as np

data = np.zeros((len(densities), len(cutoffs)))
for record in records:
    ecut = record.parameters.calculator['mode']['ecut']
    density = record.parameters.calculator['kpts']['density']
    etot = record.result.etot
    data[cutoffs.index(ecut), densities.index(density)] = etot


for i, ecut in enumerate(cutoffs):
    plt.plot(densities, data[i], label=ecut)

plt.legend()
plt.ylabel('Total energy (eV)')
plt.xlabel('K-point density (Å)')
plt.tight_layout()
plt.savefig('si-convergence.svg')
plt.show()
