from typing import Sequence

# Define calculators that are needed for the params.json file
# of each individual defect and charge folder.
# Note, that the only real change to the default relax and gs
# parameters is 'spinpol' here. Should be changed in 'master'.
relax_calc_dict = {'name': 'gpaw',
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
                   'spinpol': True}


def main(rn, atoms,
         charge_states: Sequence[int] = [0],
         calculator: dict = relax_calc_dict):
    atoms_dict = {}
    for q in charge_states:
        subfolder = f"charge_{q}"
        calculator['charge'] = q
        rn2 = rn.with_subdirectory(subfolder)
        atoms_dict[f'charge_{q}'] = rn2.task('asr.c2db.relax.main',
                                             atoms=atoms,
                                             calculator=calculator)
    return atoms_dict


class SetupAndRelaxDefects:
    def __init__(self, rn, atoms, charge_states=[0], calculator=relax_calc_dict,
                 setup_defect_kwargs={}):
        from asr.setup.defects import main as main_setup

        self.Defect_dict = main_setup(rn, atoms, **setup_defect_kwargs)
        self.relaxed_defect = {}
        for key, item in self.Defect_dict.items():
            rn2 = rn.with_subdirectory(str(item['path']))
            self.relaxed_defect[key] = main(rn, atoms=item['atoms'].output,
                                            charge_states=charge_states)
