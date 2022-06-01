from typing import Sequence


def main(rn, atoms,
         charge_states: Sequence[int] = [0],
         calculator: dict = {'name': 'gpaw',
                             'mode': {
                                 'name': 'pw',
                                 'ecut': 800,
                                 'dedecut': 'estimate'},
                             'xc': 'PBE',
                             'kpts': {
                                 'density': 6.0,
                                 'gamma': True},
                             'basis': 'dzp',
                             'symmetry': {
                                 'symmorphic': False},
                             'convergence': {
                                 'forces': 1e-4},
                             'txt': 'relax.txt',
                             'occupations': {
                                 'name': 'fermi-dirac',
                                 'width': 0.02},
                             'spinpol': True}):
    atoms_dict = {}
    if(type(charge_states) == int):
        charge_states = [charge_states]
    for q in charge_states:
        subfolder = f"charge_{q}"
        calculator['charge'] = q
        rn2 = rn.with_subdirectory(subfolder)
        atoms_dict[f'charge_{q}'] = rn2.task('asr.c2db.relax.main',
                                             atoms=atoms,
                                             calculator=calculator)
    return atoms_dict
