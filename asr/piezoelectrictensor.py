"""Piezoelectric tensor.

Module containing functionality for calculating the piezoelectric
tensor. The central recipe of this module is
:func:`asr.piezoelectrictensor.main`.

"""

import itertools
import typing

from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack

from asr.formalpolarization import main as formalpolarization
from asr.relax import main as relax

import asr

from asr.core import (
    command, option, ASRResult, prepare_result,
    calcopt, atomsopt
)
from asr.database.browser import matrixtable, make_panel_description, describe_entry


panel_description = make_panel_description("""
The piezoelectric tensor, c, is a rank-3 tensor relating the macroscopic
polarization to an applied strain. In Voigt notation, c is expressed as a 3xN
matrix relating the (x,y,z) components of the polarizability to the N
independent components of the strain tensor. The polarization in a periodic
direction is calculated as an integral over Berry phases. The polarization in a
non-periodic direction is obtained by direct evaluation of the first moment of
the electron density.
""")


all_voigt_labels = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
all_voigt_indices = [[0, 1, 2, 1, 0, 0],
                     [0, 1, 2, 2, 2, 1]]


def get_voigt_mask(pbc_c: typing.List[bool]):
    non_pbc_axes = set(char for char, pbc in zip('xyz', pbc_c) if not pbc)

    mask = [False
            if set(voigt_label).intersection(non_pbc_axes)
            else True
            for voigt_label in all_voigt_labels]
    return mask


def get_voigt_indices(pbc: typing.List[bool]):
    mask = get_voigt_mask(pbc)
    return [list(itertools.compress(indices, mask)) for indices in all_voigt_indices]


def get_voigt_labels(pbc: typing.List[bool]):
    mask = get_voigt_mask(pbc)
    return list(itertools.compress(all_voigt_labels, mask))


def webpanel(result, context):

    piezodata = result  # row.data['results-asr.piezoelectrictensor.json']
    e_vvv = piezodata['eps_vvv']
    e0_vvv = piezodata['eps_clamped_vvv']

    pbc = context.atoms.pbc

    voigt_indices = get_voigt_indices(pbc)
    voigt_labels = get_voigt_labels(pbc)

    e_ij = e_vvv[:,
                 voigt_indices[0],
                 voigt_indices[1]]
    e0_ij = e0_vvv[:,
                   voigt_indices[0],
                   voigt_indices[1]]

    etable = matrixtable(e_ij,
                         columnlabels=voigt_labels,
                         rowlabels=['x', 'y', 'z'],
                         title='c<sub>ij</sub> (e/Å<sup>dim-1</sup>)')

    e0table = matrixtable(
        e0_ij,
        columnlabels=voigt_labels,
        rowlabels=['x', 'y', 'z'],
        title='c<sup>clamped</sup><sub>ij</sub> (e/Å<sup>dim-1</sup>)')

    columns = [[etable], [e0table]]

    panel = {'title': describe_entry('Piezoelectric tensor',
                                     panel_description),
             'columns': columns}

    return [panel]


@prepare_result
class Result(ASRResult):

    eps_vvv: typing.List[typing.List[typing.List[float]]]
    eps_clamped_vvv: typing.List[typing.List[typing.List[float]]]

    key_descriptions = {'eps_vvv': 'Piezoelectric tensor.',
                        'eps_clamped_vvv': 'Piezoelectric tensor.'}
    formats = {'webpanel2': webpanel}


def convert_density_to_size(parameters):
    atoms = parameters.atoms
    calculator = parameters.calculator
    # From experience it is important to use
    # non-gamma centered grid when using symmetries.
    # Might have something to do with degeneracies, not sure.
    if 'density' in calculator['kpts']:
        kpts = calculator['kpts']
        density = kpts.pop('density')
        kpts['size'] = kptdensity2monkhorstpack(atoms, density, True)
    return parameters


sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.name = sel.EQ('asr.piezoelectrictensor')
sel.parameters = sel.NOT(sel.CONTAINS('relaxcalculator'))


@asr.migration(selector=sel)
def add_relaxcalculator_parameter(record):
    """Add relaxcalculator parameter and delete unused dependency parameters."""
    dep_params = record.parameters.dependency_parameters
    record.parameters.relaxcalculator = dep_params['asr.relax']['calculator']
    del_par = {'calculator', 'd3',
               'allow_symmetry_breaking', 'fixcell'}
    for par in del_par:
        del dep_params['asr.relax'][par]

    del_par = {'gpwname'}
    for par in del_par:
        del dep_params['asr.formalpolarization'][par]

    if 'calculator' in record.parameters:
        del dep_params['asr.formalpolarization']['calculator']
    return record


@command(
    module="asr.piezoelectrictensor",
    argument_hooks=[convert_density_to_size],
)
@atomsopt
@option('--strain-percent', help='Strain fraction.', type=float)
@calcopt
@asr.calcopt(aliases=['--relaxcalculator'], help='Calculator parameters.')
def main(
        atoms: Atoms,
        strain_percent: float = 1,
        calculator: dict = formalpolarization.defaults.calculator,
        relaxcalculator: dict = relax.defaults.calculator,
) -> Result:
    """Calculate piezoelectric tensor.

    This recipe calculates the clamped and full piezoelectric
    tensor. You generally will only need the full piezoelectric
    tensor. The clamped piezoelectric tensor is useful for analyzing
    results. The piezoelectric tensor is calculated using a finite
    difference scheme by calculating the derivative of the
    polarization density at finite strains.

    Parameters
    ----------
    strain_percent : float
        Amount of strain applied to the material.
    calculator : dict
        Calculator parameters.

    """
    import numpy as np
    from ase.units import Bohr
    from asr.setup.strains import main as make_strained_atoms
    from asr.setup.strains import get_relevant_strains
    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    pbc_c = atoms.get_pbc()
    if not all(pbc_c):
        N = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
    else:
        N = 1.0
    eps_clamped_vvv = np.zeros((3, 3, 3), float)
    eps_vvv = np.zeros((3, 3, 3), float)
    ij = get_relevant_strains(atoms.pbc)

    for clamped in [True, False]:
        for i, j in ij:
            phase_sc = np.zeros((2, 3), float)
            for s, sign in enumerate([-1, 1]):
                strained_atoms = make_strained_atoms(
                    atoms,
                    strain_percent=sign * strain_percent,
                    i=i, j=j)

                if clamped:
                    atoms_for_pol = strained_atoms
                else:
                    relaxres = relax(
                        atoms=strained_atoms,
                        calculator=relaxcalculator,
                        fixcell=True,
                        d3=False,
                        allow_symmetry_breaking=True,
                    )
                    atoms_for_pol = relaxres.atoms

                polresults = formalpolarization(
                    atoms=atoms_for_pol,
                    calculator=calculator,
                )

                phase_sc[s] = polresults['phase_c']

            dphase_c = phase_sc[1] - phase_sc[0]
            dphase_c -= np.round(dphase_c / (2 * np.pi)) * 2 * np.pi
            dphasedeps_c = dphase_c / (2 * strain_percent * 0.01)
            eps_v = (np.dot(dphasedeps_c, cell_cv)
                     / (2 * np.pi * vol))
            eps_v *= N

            if clamped:
                epsref_vvv = eps_clamped_vvv
            else:
                epsref_vvv = eps_vvv

            epsref_vvv[:, i, j] = eps_v
            epsref_vvv[:, j, i] = eps_v

    data = {'eps_vvv': eps_vvv,
            'eps_clamped_vvv': eps_clamped_vvv}

    return Result(data=data)


if __name__ == '__main__':
    main.cli()
