"""Stiffness tensor."""
import typing

import numpy as np

import asr
from asr.core import ASRResult, prepare_result
from asr.database.browser import (matrixtable, describe_entry, dl,
                                  make_panel_description)
from asr.c2db.relax import main as relax

panel_description = make_panel_description(
    """
The stiffness tensor (C) is a rank-4 tensor that relates the stress of a
material to the applied strain. In Voigt notation, C is expressed as a NxN
matrix relating the N independent components of the stress and strain
tensors. C is calculated as a finite difference of the stress under an applied
strain with full relaxation of atomic coordinates. A negative eigenvalue of C
indicates a dynamical instability.
""",
    articles=['C2DB'],
)


def webpanel(result, context):
    stiffnessdata = result  # row.data['results-asr.c2db.stiffness.json']
    c_ij = stiffnessdata['stiffness_tensor'].copy()
    eigs = stiffnessdata['eigenvalues'].copy()
    nd = context.ndim

    if nd == 2:
        c_ij = np.zeros((4, 4))
        c_ij[1:, 1:] = stiffnessdata['stiffness_tensor']
        ctable = matrixtable(
            stiffnessdata['stiffness_tensor'],
            title='C<sub>ij</sub> (N/m)',
            columnlabels=['xx', 'yy', 'xy'],
            rowlabels=['xx', 'yy', 'xy'])

        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue {ie}', f'{eig.real:.2f} N/m']
                      for ie, eig in enumerate(sorted(eigs,
                                                      key=lambda x: x.real))])
    elif nd == 3:
        eigs *= 1e-9
        c_ij *= 1e-9
        ctable = matrixtable(
            c_ij,
            title='C<sub>ij</sub> (10⁹ N/m²)',
            columnlabels=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'],
            rowlabels=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'])

        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue {ie}', f'{eig.real:.2f} · 10⁹ N/m²']
                      for ie, eig
                      in enumerate(sorted(eigs, key=lambda x: x.real))])
    else:
        ctable = dict(
            type='table',
            rows=[])
        eig = complex(eigs[0])
        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [['Eigenvalue', f'{eig.real:.2f} * 10⁻¹⁰ N']])

    eigtable = dict(
        type='table',
        rows=eigrows)

    panel = {
        'title': describe_entry(
            'Stiffness tensor', description=panel_description
        ),
        'columns': [[ctable], [eigtable]],
        'sort': 2}

    dynstab = result['dynamic_stability_stiffness']

    row = [
        describe_entry(
            'Dynamical (stiffness)',
            'Classifier for the dynamical stability of a material '
            'based on the minimum eigenvalue of the stiffness tensor.'
            + dl(
                [
                    ["LOW", dynstab_text_low],
                    ["HIGH", dynstab_text_high],
                ]
            )
        ),
        dynstab.upper()]

    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Stability', 'Category'],
                             'rows': [row],
                             }]],
               'sort': 3}

    return [panel, summary]


@prepare_result
class Result(ASRResult):

    c_11: float
    c_12: float
    c_13: float
    c_14: float
    c_15: float
    c_16: float
    c_21: float
    c_22: float
    c_23: float
    c_24: float
    c_25: float
    c_26: float
    c_31: float
    c_32: float
    c_33: float
    c_34: float
    c_35: float
    c_36: float
    c_41: float
    c_42: float
    c_43: float
    c_44: float
    c_45: float
    c_46: float
    c_51: float
    c_52: float
    c_53: float
    c_54: float
    c_55: float
    c_56: float
    c_61: float
    c_62: float
    c_63: float
    c_64: float
    c_65: float
    c_66: float

    stiffness_tensor: typing.List[typing.List[float]]
    eigenvalues: typing.List[complex]
    dynamic_stability_stiffness: str
    speed_of_sound_x: float
    speed_of_sound_y: float

    key_descriptions = {
        "c_11": "Stiffness tensor 11-component.",
        "c_12": "Stiffness tensor 12-component.",
        "c_13": "Stiffness tensor 13-component.",
        "c_14": "Stiffness tensor 14-component.",
        "c_15": "Stiffness tensor 15-component.",
        "c_16": "Stiffness tensor 16-component.",
        "c_21": "Stiffness tensor 21-component.",
        "c_22": "Stiffness tensor 22-component.",
        "c_23": "Stiffness tensor 23-component.",
        "c_24": "Stiffness tensor 24-component.",
        "c_25": "Stiffness tensor 25-component.",
        "c_26": "Stiffness tensor 26-component.",
        "c_31": "Stiffness tensor 31-component.",
        "c_32": "Stiffness tensor 32-component.",
        "c_33": "Stiffness tensor 33-component.",
        "c_34": "Stiffness tensor 34-component.",
        "c_35": "Stiffness tensor 35-component.",
        "c_36": "Stiffness tensor 36-component.",
        "c_41": "Stiffness tensor 41-component.",
        "c_42": "Stiffness tensor 42-component.",
        "c_43": "Stiffness tensor 43-component.",
        "c_44": "Stiffness tensor 44-component.",
        "c_45": "Stiffness tensor 45-component.",
        "c_46": "Stiffness tensor 46-component.",
        "c_51": "Stiffness tensor 51-component.",
        "c_52": "Stiffness tensor 52-component.",
        "c_53": "Stiffness tensor 53-component.",
        "c_54": "Stiffness tensor 54-component.",
        "c_55": "Stiffness tensor 55-component.",
        "c_56": "Stiffness tensor 56-component.",
        "c_61": "Stiffness tensor 61-component.",
        "c_62": "Stiffness tensor 62-component.",
        "c_63": "Stiffness tensor 63-component.",
        "c_64": "Stiffness tensor 64-component.",
        "c_65": "Stiffness tensor 65-component.",
        "c_66": "Stiffness tensor 66-component.",
        "eigenvalues": "Stiffness tensor eigenvalues.",
        "speed_of_sound_x": "Speed of sound (x) [m/s]",
        "speed_of_sound_y": "Speed of sound (y) [m/s]",
        "stiffness_tensor": "Stiffness tensor [`N/m^{dim-1}`]",
        "dynamic_stability_stiffness":
        "Stiffness dynamic stability (low/high)",
    }

    formats = {'webpanel2': webpanel}


sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.parameters.dependency_parameters = sel.CONTAINS("asr.c2db.relax:main")


@asr.mutation(selector=sel)
def transform_stiffness_resultfile_record(record):
    """Remove fixcell and allow_symmetry_breaking from dependency_parameters."""
    dep_params = record.parameters['dependency_parameters']
    relax_dep_params = dep_params['asr.c2db.relax:main']
    delparams = {
        'fixcell',
        'allow_symmetry_breaking',
        'atoms',
        'tmp_atoms',
        'tmp_atoms_file',
    }
    for param in delparams:
        if param in relax_dep_params:
            del relax_dep_params[param]
    if "fmax" not in relax_dep_params:
        record.parameters.fmax = 0.01  # 0.01 is a historical constant
    if "enforce_symmetry" not in relax_dep_params:
        record.parameters.enforce_symmetry = True  # True because of history
    return record


# @command(
#     module='asr.c2db.stiffness',
# )
# @option('--atoms', type=AtomsFile(), help='Atoms to be strained.',
#         default='structure.json')
# @asr.calcopt
# @option('--strain-percent', help='Magnitude of applied strain.', type=float)
# @option('--d3/--nod3', help='Relax with vdW D3.', is_flag=True)
# @option('--fmax', help='Maximum force allowed.', type=float)
# @option('--enforce-symmetry/--dont-enforce-symmetry',
#         help='Symmetrize forces and stresses.', is_flag=True)


class StrainWorkflow:
    def __init__(self, rn, atoms, strain_percent: float = 1.0):
        from asr.setup.strains import get_relevant_strains
        self.strains = []
        # (If atoms is a future, we cannot refer to atoms.pbc.)
        ij = get_relevant_strains(atoms.pbc)

        self.strain_percent = strain_percent
        self.atoms = atoms
        self.strains = {}

        for i, j in ij:
            for sign in [-1, 1]:
                key = StrainID(i, j, sign)
                strained = rn.task(
                    'asr.setup.strains.main',
                    name=key.name,
                    atoms=atoms,
                    # XXX dangerous floating point multiplication:
                    strain_percent=sign * strain_percent,
                    i=i, j=j)
                self.strains[key] = strained


from collections import namedtuple
class StrainID(namedtuple('StrainID', ['i', 'j', 'sign'])):
    @property
    def name(self):
        return f'{self.i}_{self.j}_{self.sign}'


def stiffnesstensor(stress_tensors, strain_percent):
    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    stiffness_tensor = np.zeros((6, 6))

    for (i, j, sign), stress_tensor in stress_tensors:
        stiffness_index = ij_to_voigt[i][j]
        stiffness_tensor[:, stiffness_index] += sign * stress_tensor

    stiffness_tensor /= strain_percent * 0.02
    return stiffness_tensor


class StiffnessWorkflow:
    def __init__(
            self,
            rn,
            strainworkflow,
            calculator: dict = relax.defaults.calculator,
            d3: bool = False,
            fmax: float = relax.defaults.fmax,
            enforce_symmetry: bool = True):
        """Calculate stiffness tensor."""

        atoms = strainworkflow.atoms

        self.relaxations = {}
        for key, strained in strainworkflow.strains.items():
            self.relaxations[key] = rn.task(
                'asr.c2db.relax.main',
                name=f'strainrelax-{key.name}',
                atoms=strained.output,
                calculator=calculator,
                fixcell=True,
                allow_symmetry_breaking=True,
                d3=d3,
                fmax=fmax,
                enforce_symmetry=enforce_symmetry)

        # Due to JSON key restrictions we flatten the relaxations dictionary:
        stress_tensors_flat = [[key, relaxation.output['stress']]
                               for key, relaxation
                               in self.relaxations.items()]

        self.stiffness = rn.task(
            'asr.c2db.stiffness.stiffnesstensor',
            name='stiffnesstensor',
            stress_tensors=stress_tensors_flat,
            strain_percent=strainworkflow.strain_percent)

        self.postprocess = rn.task(
            'asr.c2db.stiffness.postprocess',
            name='stiffnesspostprocess',
            atoms=atoms,
            stiffness=self.stiffness.output)


def postprocess(atoms, stiffness):
    from ase.units import J

    # We work with Mandel notation which is conventional and convenient
    stiffness[3:, :] *= 2**0.5
    stiffness[:, 3:] *= 2**0.5

    # Convert the stiffness tensor from [eV/Ang^3] -> [J/m^3]=[N/m^2]
    stiffness *= 10**30 / J

    # Now do some post processing
    data = {}
    nd = sum(atoms.pbc)
    speed_of_sound_x = None
    speed_of_sound_y = None
    if nd == 2:
        cell = atoms.get_cell()
        # We have to normalize with the supercell size
        z = cell[2, 2]
        stiffness = stiffness[[0, 1, 5], :][:, [0, 1, 5]] * z * 1e-10
        from ase.units import kg
        from ase.units import m as meter
        area = atoms.get_volume() / cell[2, 2]
        mass = sum(atoms.get_masses())
        area_density = (mass / kg) / (area / meter**2)
        # speed of sound in m/s
        speed_of_sound_x = np.sqrt(stiffness[0, 0] / area_density)
        speed_of_sound_y = np.sqrt(stiffness[1, 1] / area_density)
    elif nd == 1:
        cell = atoms.get_cell()
        area = atoms.get_volume() / cell[2, 2]
        stiffness = stiffness[[2], :][:, [2]] * area * 1e-20
        # typical values for 1D are of the order of 10^(-10) N
    elif nd == 0:
        raise RuntimeError('Cannot compute stiffness tensor of 0D material.')

    data['speed_of_sound_x'] = speed_of_sound_x
    data['speed_of_sound_y'] = speed_of_sound_y

    for i in range(6):
        for j in range(6):
            data[f'c_{i + 1}{j + 1}'] = None
    stiffness_shape = stiffness.shape
    for i in range(stiffness_shape[0]):
        for j in range(stiffness_shape[1]):
            data[f'c_{i + 1}{j + 1}'] = stiffness[i, j]

    data['stiffness_tensor'] = stiffness

    if nd == 1:
        eigs = stiffness
    else:
        eigs = np.linalg.eigvals(stiffness)
    data['eigenvalues'] = eigs

    data['dynamic_stability_stiffness'] = dynamic_stability_stiffness(
        eigs.min())
    return Result(data=data)


dynstab_text_high = 'Minimum stiffness tensor eigenvalue > 0'
dynstab_text_low = 'Minimum stiffness tensor eigenvalue ≤ 0'


def dynamic_stability_stiffness(mineig):
    if mineig > 0:
        return 'high'
    else:
        return 'low'
