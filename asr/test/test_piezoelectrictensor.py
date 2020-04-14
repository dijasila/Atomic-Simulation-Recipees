from ase.units import Bohr
import numpy as np
import pytest


def zero_pad_non_pbc_strain_directions(matrix_cvv, pbc_c):
    return matrix_cvv * pbc_c[None] * pbc_c[None, None]


def get_strain_from_atoms(inv_cell_vc, atoms):
    return np.dot(inv_cell_vc, atoms.get_cell() / Bohr) - np.eye(3)


@pytest.mark.ci
@pytest.mark.parametrize("nspins", [1, 2])
def test_piezoelectrictensor(asr_tmpdir_w_params, mockgpaw, mocker, test_material,
                             nspins, get_webcontent):
    import numpy as np
    from gpaw import GPAW

    cell_cv = test_material.get_cell() / Bohr
    pbc_c = test_material.get_pbc()
    inv_cell_vc = np.linalg.inv(cell_cv)
    # dphase_c / dstrain_vv
    dpde_cvv = zero_pad_non_pbc_strain_directions(
        np.array([np.eye(3)] * 3),
        pbc_c,
    )

    # ddipol_v / dstrain_vv
    ddipolde_vvv = zero_pad_non_pbc_strain_directions(
        np.array([np.eye(3)] * 3),
        pbc_c,
    )

    # Also move atomic positions
    natoms = len(test_material)
    dsposde_acvv = np.zeros((natoms, 3, 3, 3), float)
    dsposde_acvv[:, :] = np.eye(3)
    dsposde_acvv *= pbc_c[None, None] * pbc_c[None, None, None]
    spos_ac = test_material.get_scaled_positions(wrap=True)

    def _get_berry_phases(self, dir=0, spin=0):
        strain_vv = get_strain_from_atoms(inv_cell_vc, self.atoms)
        phase_c = np.dot(dpde_cvv.reshape(3, -1), strain_vv.reshape(-1))
        return [phase_c[dir]]

    def _get_scaled_positions(self):
        strain_vv = get_strain_from_atoms(inv_cell_vc, self.atoms)
        dspos_ac = np.dot(dsposde_acvv.reshape(natoms, 3, -1),
                          strain_vv.reshape(-1))
        return spos_ac + dspos_ac

    def _get_dipole_moment(self):
        strain_vv = get_strain_from_atoms(inv_cell_vc, self.atoms)
        dipol_v = np.dot(ddipolde_vvv.reshape(3, -1), strain_vv.reshape(-1))
        return dipol_v

    def _get_setup_nvalence(self, element_number):
        return 1

    def get_number_of_spins(self):
        return nspins

    mocker.patch.object(GPAW, '_get_berry_phases', new=_get_berry_phases)
    mocker.patch.object(GPAW, '_get_dipole_moment', new=_get_dipole_moment)
    mocker.patch.object(GPAW, '_get_scaled_positions', new=_get_scaled_positions)
    mocker.patch.object(GPAW, '_get_setup_nvalence', new=_get_setup_nvalence)
    mocker.patch.object(GPAW, 'get_number_of_spins', new=get_number_of_spins)
    from ase.io import write
    from asr.piezoelectrictensor import main
    write('structure.json', test_material)
    results = main()
    content = get_webcontent('database.db')

    N = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
    vol = test_material.get_volume() / Bohr**3

    # Formula for piezoeletric tensor
    # (The last factor of 2 is for spins)
    eps_berry_analytic_vvv = np.tensordot(
        cell_cv, dpde_cvv,
        axes=([0], [0])
    ) / (2 * np.pi * vol) * N * 2

    eps_dipole_analytic_vvv = ddipolde_vvv / (vol * Bohr) * N

    Z_a = [1] * natoms
    eps_atomic_analytic_vvv = np.tensordot(
        cell_cv,
        np.tensordot(
            Z_a,
            dsposde_acvv,
            axes=([0], [0]),
        ),
        axes=([0], [0])
    ) / vol * N

    eps_analytic_vvv = np.zeros((3, 3, 3), float)
    eps_analytic_vvv[pbc_c] = (eps_berry_analytic_vvv[pbc_c]
                               + eps_atomic_analytic_vvv[pbc_c])
    eps_analytic_vvv[~pbc_c] = eps_dipole_analytic_vvv[~pbc_c]

    eps_vvv = results['eps_vvv']
    eps_clamped_vvv = results['eps_clamped_vvv']
    assert eps_vvv == pytest.approx(eps_analytic_vvv)
    assert eps_clamped_vvv == pytest.approx(eps_analytic_vvv)
    assert "Piezoelectrictensor" in content, content


@pytest.mark.acceptance_test
def test_piezo_BN(asr_tmpdir_w_params):
    from .materials import BN
    from asr.piezoelectrictensor import main
    from asr.setup.params import main as setupparams

    formal_calculator = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 300},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 2.0},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'txt': 'formalpol.txt',
        'charge': 0
    }
    setupparams(params={'asr.formalpolarization': {'calculator': formal_calculator}})
    BN.write('structure.json')
    results = main()

    full_response = 0.12  # From a default parameter calculation of BN
    clamped_response = 0.044  # From a default parameter calculation of BN
    for eps, response in [('eps_vvv', full_response),
                          ('eps_clamped_vvv', clamped_response)]:
        eps_calculated_vvv = results[eps]
        eps_converged_vvv = np.array([[[0.0, response, 0.0],
                                       [response, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]],
                                      [[response, 0.0, 0.0],
                                       [0.0, -response, 0.0],
                                       [0.0, 0.0, 0.0]],
                                      [[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]]])

        assert eps_calculated_vvv == pytest.approx(eps_converged_vvv, rel=0.05)
