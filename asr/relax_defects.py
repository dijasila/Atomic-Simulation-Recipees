from typing import Sequence
from gpaw.atom.generator2 import get_number_of_electrons


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

gs_calc_dict = {'name': 'gpaw',
                'mode': {'name': 'pw', 'ecut': 800},
                'xc': 'PBE',
                'basis': 'dzp',
                'kpts': {'density': 12.0, 'gamma': True},
                'occupations': {'name': 'fermi-dirac',
                                'width': 0.02},
                'convergence': {'bands': 'CBM+3.0'},
                'nbands': '200%',
                'txt': 'gs.txt',
                'spinpol': True}


# Counts number of electrons and returns True if even number and False if odd number
def count_e(atoms):
    from asr.setup.defects import ref_to_atoms
    atoms1 = ref_to_atoms(atoms)
    Ne = sum([get_number_of_electrons(atom.symbol, name=None) for atom in atoms1])
    print('  Number of electrons: ', Ne)
    if Ne % 2 == 0:
        Neven = True
    else:
        Neven = False
    return Neven


def relax_defects(rn, atoms,
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


def gs_defects(rn, relax_dict,
               calculator: dict = gs_calc_dict):
    scf = {}
    for key, item in relax_dict.items():
        rn2 = rn.with_subdirectory(key)
        scf[key] = rn2.task('asr.c2db.gs.calculate',
                            atoms=item)
    return scf


class SetupAndRelaxDefects:
    def __init__(self, rn, atoms, charge_states=[0],
                 calculator=relax_calc_dict,
                 setup_defect_kwargs={}):
        from asr.setup.defects import main as main_setup
        """
        atoms: host atoms object
        charge_states: Sequence of integer numbers defining charge states
        calculator: dict defining calculator for relax calculation
        setup_defect_kwargs: dict including any changes to default
                             input arguments for setup.defect.main
        """

        self.Defect_dict = main_setup(rn, atoms, **setup_defect_kwargs)
        self.relaxed_defect = {}
        for key, item in self.Defect_dict.items():
            rn2 = rn.with_subdirectory(key)
            self.relaxed_defect[key] = relax_defects(rn2, atoms=item.output,
                                                     charge_states=charge_states)
