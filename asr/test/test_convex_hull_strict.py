import pytest
from asr.c2db.convex_hull import (calculate_hof_and_hull, LOW, MEDIUM, HIGH,
                                  Reference)


@pytest.mark.ci
def test_single_species():
    ref = mkref('A', 1, 0.0)
    energy = 42.0
    result = calculate_hof_and_hull('A', energy, [ref], {'A': energy})
    assert result['hform'] == pytest.approx(0.0)
    assert result['ehull'] == pytest.approx(0.0)
    assert result['thermodynamic_stability_level'] == HIGH
    assert result['indices'] == [0]
    assert result['coefs'] == [pytest.approx(1.0)]


def mkref(formula, natoms, hform):
    # len(formula) may not be natoms due to reduced formula!
    return Reference(formula=formula, natoms=natoms,
                     hform_per_atom=hform)


def stdrefs():
    return [mkref('Ga', 4, 0.0), mkref('As', 2, 0.0)]


def refs_with_gaas():
    return [*stdrefs(), mkref('GaAs', 2, -0.348)]


# Below we test that the classification of LOW/MEDIUM/HIGH
# works as expected in certain cases.


e0_gaas = -8.263  # this is for both atoms
hform_gaas = -0.348  # this is per atom
ref_energies_per_atom = {'Ga': -2.903, 'As': -4.663}
ref_energy_sum = sum(ref_energies_per_atom.values())

natoms = 2
stab_tolerance = 0.2  # Tolerance for accepting materials as stable in recipe

delta_e_high = -0.5
delta_e_medium = stab_tolerance + 0.05


# * First we test that GaAs is stable with respect to Ga and As.
# * Then we test that if we lower the energy of GaAs with respect to Ga and As,
#   then the ehull and hform decrease consistently with that.  In this and
#   subsequent tests, the real GaAs reference is included, such that we are
#   really adding fictitious "new" GaAs phases to be compared with Ga, As, and
#   standard GaAs.
# * Then we test "medium" stability
# * Then "low"
#
# Note: The logic to establish the right referen
@pytest.mark.ci
@pytest.mark.parametrize(
    'energy, refs, hformref, e_over_hull_ref, rating',
    [(e0_gaas, stdrefs(), hform_gaas, hform_gaas, HIGH),
     (e0_gaas + natoms * delta_e_high, refs_with_gaas(),
      hform_gaas + delta_e_high, delta_e_high, HIGH),
     (e0_gaas + natoms * delta_e_medium, refs_with_gaas(),
      hform_gaas + delta_e_medium, delta_e_medium, MEDIUM),
     (ref_energy_sum + natoms * stab_tolerance + 1e-10, refs_with_gaas(),
      stab_tolerance, -hform_gaas + stab_tolerance, LOW)])
def test(energy, refs, hformref, e_over_hull_ref, rating):
    formula = 'GaAs'

    result = calculate_hof_and_hull(
        formula, energy, refs, ref_energies_per_atom)

    assert result['hform'] == pytest.approx(hformref, abs=0.001)
    assert result['ehull'] == pytest.approx(e_over_hull_ref, abs=0.001)
    assert result['thermodynamic_stability_level'] == rating
