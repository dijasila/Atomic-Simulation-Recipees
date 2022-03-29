"""Utility module for handling symmetries."""


def _atoms2symmetry_gpaw(atoms, id_a=None, tolerance=1e-7):
    """Create gpaw.symmetry.Symmetry object from atoms object.

    Note this can be substituted with gpaw.symmetry.atoms2symmetry in
    a future gpaw release.

    """
    from gpaw.symmetry import Symmetry
    if id_a is None:
        id_a = atoms.get_atomic_numbers()
    symmetry = Symmetry(id_a, atoms.cell, atoms.pbc,
                        symmorphic=False,
                        time_reversal=False,
                        tolerance=tolerance)
    symmetry.analyze(atoms.get_scaled_positions())
    return symmetry


def atoms2spgcell(atoms, magmoms=None):
    """Convert an ase.Atoms object to spglib cell."""
    lattice = atoms.get_cell().array
    positions = atoms.get_scaled_positions(wrap=False)
    numbers = atoms.get_atomic_numbers()
    return (lattice, positions, numbers)


def has_inversion(rotations_n):
    """Determine if atoms has inversion symmetry.

    Parameters
    ----------
    rotations_n : list of 3x3 matrices
        List of point group symmetries.

    Returns
    -------
    bool
        True of rotations_n contains inversion symmetry.
    """
    import numpy as np
    inversion = -np.identity(3, dtype=int)
    return np.any([np.all(rotation == inversion) for rotation in rotations_n])


def atoms2symmetry(atoms, tolerance=1e-7, angle_tolerance=0.01):
    """Return a SimpleNamespace containing symmetries.

    Uses spglib to determine symmetries.

    Parameters
    ----------
    atoms : `ase.Atoms` object
    tolerance : float
        spglib symmetry tolerance.
    angle_tolerance : float
        spglib angle tolerance.

    Returns
    -------
    symmetry : SimpleNamespace
        dataset : dict
            spglib dataset
        has_inversion : bool
            Does structure have an inversion center?
    """
    from types import SimpleNamespace
    from spglib import get_symmetry_dataset

    cell = atoms2spgcell(atoms)
    dataset = get_symmetry_dataset(cell, symprec=tolerance,
                                   angle_tolerance=angle_tolerance)

    symmetry = SimpleNamespace(
        has_inversion=has_inversion(dataset['rotations']),
        dataset=dataset
    )
    return symmetry


def restrict_spin_projection_2d(kpt, op_scc, s_vm):
    """Restrict spin projections according to symmetry."""
    import numpy as np
    mirror_count = 0
    s_vm = s_vm.copy()
    for symm in op_scc:
        # Inversion symmetry forces spin degeneracy and 180 degree rotation
        # forces the spins to lie in plane
        if np.allclose(symm, -1 * np.eye(3)):
            s_vm[:] = 0
            continue
        vals, vecs = np.linalg.eigh(symm)
        # A mirror plane
        if np.allclose(np.abs(vals), 1) and np.allclose(np.prod(vals), -1):
            # Mapping k -> k, modulo a lattice vector
            if np.allclose(kpt % 1, (np.dot(symm, kpt)) % 1):
                mirror_count += 1
    # If we have two or more mirror planes,
    # then we must have a spin-degenerate
    # subspace
    if mirror_count >= 2:
        s_vm[2, :] = 0

    return s_vm


# In C2DB, the relaxation is not so strict that we can use a very strict
# symmetry criterion.  Therefore we have a hardcoded default here
# that recipes can use.
#
# Ideally the criterion would probably depend on the Hessian or stiffness,
# but 0.1 and 1.0 works reasonably well according to tests by Thomas Olsen.

c2db_symmetry_eps = 0.1
c2db_symmetry_angle = 1.0
