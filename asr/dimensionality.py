from asr.core import read_json
from ase.geometry.dimensionality import analyze_dimensionality


atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
atoms.cell[2, 2] = 7.0
atoms.set_pbc((1, 1, 1))
atoms *= 3

intervals = analyze_dimensionality(atoms)
m = intervals[0]
print(sum([e.score for e in intervals]))
print(m.dimtype, m.h, m.score, m.a, m.b)

atoms.set_tags(m.components)
for interval in intervals:
    print(interval)



def main(structure):
    intervals = analyze_dimensionality()
