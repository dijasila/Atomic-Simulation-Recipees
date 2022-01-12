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
        'evac': 0}}
    ]


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


@pytest.mark.ci
def test_ordered_transitions():
    from asr.sj_analyze import order_transition

    new = {'transition_name': '2/3',
           'transition_values': {
               'transition': 1,
               'erelax': 0,
               'evac': 0}}
    new_transitions = [transitions[1], new, transitions[0]]
