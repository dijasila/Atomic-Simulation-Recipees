"""Test Slater Janak recipe."""
import pytest

transitions = [{
    'transition_name': '0/1',
    'transition_values': {
        'transition': 0.5,
        'erelax': 0,
        'evac': 0}},
    {
    'transition_name': '0/-1',
    'transition_values': {
        'transition': 1,
        'erelax': 0,
        'evac': 0}}]


@pytest.mark.parametrize('tran', [[transitions[0]], transitions[1]])
@pytest.mark.ci
def test_calculate_formation_energies(tran):
    from asr.sj_analyze import calculate_formation_energies
    vbm = 0.4
    eform_0 = 1
    results = {
        '0': eform_0,
        '1': (eform_0
              + (vbm
                 - transitions[0]['transition_values']['transition']) * 1),
        '-1': (eform_0
               + (vbm
                  - transitions[1]['transition_values']['transition']) * -1)}
    eforms = calculate_formation_energies(eform_0, transitions, vbm)

    for eform in eforms:
        assert eform[0] == pytest.approx(results[f'{str(eform[1])}'])


@pytest.mark.parametrize('ref_trans', ['1/2', '2/3', '-1/-2'])
@pytest.mark.ci
def test_ordered_transitions(ref_trans):
    from asr.sj_analyze import order_transitions

    new = {'transition_name': ref_trans,
           'transition_values': {
               'transition': 1,
               'erelax': 0,
               'evac': 0}}
    new_transitions = [transitions[1], new, transitions[0]]
    if ref_trans == '1/2' or ref_trans == '2/3':
        order = ['0/1', ref_trans, '0/-1']
    else:
        order = ['0/1', '0/-1', ref_trans]

    ordered_transitions = order_transitions(new_transitions)
    for i, transition in enumerate(ordered_transitions):
        assert transition['transition_name'] == order[i]


@pytest.mark.parametrize('hof', [0.1, -1.1])
@pytest.mark.ci
def test_get_heat_of_formation(asr_tmpdir, hof):
    from ase.db import connect
    from ase.build import mx2
    from asr.database.material_fingerprint import get_uid_of_atoms, get_hash_of_atoms
    from asr.sj_analyze import get_heat_of_formation

    db = connect('database.db')
    atoms = mx2('MoS2')
    hash = get_hash_of_atoms(atoms)
    uid = get_uid_of_atoms(atoms, hash)
    db.write(atoms, hform=hof, uid=uid)

    ref_hof = get_heat_of_formation(db, atoms)
    assert ref_hof == pytest.approx(hof)


@pytest.mark.parametrize('symbol', ['v', 'Ag'])
@pytest.mark.ci
def test_obtain_chemical_potential(asr_tmpdir, symbol):
    from ase.db import connect
    from .materials import Ag
    from ase.calculators.emt import EMT
    from asr.sj_analyze import obtain_chemical_potential

    atoms = Ag.copy()
    calc = EMT()
    atoms.calc = calc
    en = atoms.get_potential_energy()
    db = connect('database.db')
    db.write(atoms, ns=1)

    res_standard_states = obtain_chemical_potential(symbol, db)
    if symbol == 'v':
        assert res_standard_states['eref'] == pytest.approx(0)
    else:
        assert res_standard_states['eref'] == pytest.approx(en)
    assert res_standard_states['element'] == symbol
