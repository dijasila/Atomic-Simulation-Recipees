import typing
import click
import numpy as np
from pathlib import Path
from gpaw import restart
from asr.core import command, option, ASRResult, prepare_result


@prepare_result
class WaveFunctionResult(ASRResult):
    """Container for results of specific wavefunction for one spin channel."""

    state: int
    spin: int
    energy: float

    key_descriptions: typing.Dict[str, str] = dict(
        state='State index.',
        spin='Spin index (0 or 1).',
        energy='Energy of the state (ref. to the vacuum level in 2D) [eV].')


@prepare_result
class Result(ASRResult):
    """Container for asr.get_wfs results."""

    wfs: typing.List[WaveFunctionResult]
    above_below: typing.Tuple[bool, bool]
    eref: float

    key_descriptions: typing.Dict[str, str] = dict(
        wfs='List of WaveFunctionResult objects for all states.',
        above_below='States within the gap above and below EF? '
                    '(ONLY for defect systems).',
        eref='Energy reference (vacuum level in 2D, 0 otherwise) [eV].')


@command(module='asr.get_wfs',
         requires=['gs.gpw', 'structure.json'],
         dependencies=['asr.gs'],
         resources='1:10m',
         returns=Result)
@option('--state', help='Specify state index that you want to '
        'write out. This option will not be used when '
        '"--get-gapstates" is used.', type=int)
@option('--erange', help='Specify the energy range (wrt. Fermi level) '
        'within which you want to write out the wavefunctions (lower '
        'threshold, upper threshold). This option will not be used when '
        '"--get-gapstates" is used.', nargs=2,
        type=click.Tuple([float, float]))
@option('--get-gapstates/-dont-get-gapstates', help='Save all wfs '
        'for states present inside the gap (only use this option for '
        'defect systems, where your folder structure has been set up '
        'with asr.setup.defects).', is_flag=True)
def main(state: int = 0,
         erange: typing.Tuple[float, float] = (0, 0),
         get_gapstates: bool = False) -> Result:
    """
    Perform fixed density calculation and write out wavefunctions.

    This recipe reads in an existing gs.gpw file, runs a fixed density
    calculation on it, and writes out wavefunctions for a state in
    "wf.{bandindex}_{spin}.cube" cube files. This state can either be
    a particular state given by the bandindex with the option "--state",
    or (for defect systems) be all of the states in the gap which will
    be evaluated with the option "--get-gapstates". Note, that when
    "--get-gapstates" is given, "--state" will be ignored.
    """
    from gpaw import restart
    from ase.io import write
    from asr.core import read_json

    # read in converged gs.gpw file and run fixed density calculation
    print('INFO: run fixed density calculation.')
    atoms, calc = restart('gs.gpw', txt='get_wfs.txt')
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})
    # evaluate states in the gap (if '--get-gapstates' is active)
    if get_gapstates:
        print('INFO: evaluate gapstates.')
        states, above_below, eref = return_gapstates(calc)
    # get energy reference and convert given input state to correct format
    elif not get_gapstates:
        if np.sum(atoms.get_pbc()) == 2:
            eref = read_json('results-asr.gs.json')['evac']
        else:
            eref = 0
        if erange == (0, 0):
            states = [state]
        elif erange != (0, 0):
            states = return_erange_states(calc, erange)
        above_below = (None, None)

    print(f'INFO: states to write to file: {states}.')

    # loop over all states and write the wavefunctions to file,
    # set up WaveFunctionResults
    wfs_results = []
    for state in states:
        wf = calc.get_pseudo_wave_function(band=state, spin=0)
        energy = calc.get_eigenvalues(spin=0)[state] - eref
        fname = f'wf.{state}_0.cube'
        write(fname, atoms, data=wf)
        wfs_results.append(WaveFunctionResult.fromdata(
            state=state,
            spin=0,
            energy=energy))
        if calc.get_number_of_spins() == 2:
            wf = calc.get_pseudo_wave_function(band=state, spin=1)
            energy = calc.get_eigenvalues(spin=1)[state] - eref
            fname = f'wf.{state}_1.cube'
            write(fname, atoms, data=wf)
            wfs_results.append(WaveFunctionResult.fromdata(
                state=state,
                spin=1,
                energy=energy))

    return Result.fromdata(
        wfs=wfs_results,
        above_below=above_below,
        eref=eref)


def return_gapstates(calc):
    """
    Evaluate states within the pristine bandgap and return bandindices.

    This function compares a defect calculation to a pristine one and
    evaluates the gap states by aligning semi-core states of pristine and
    defect system and afterwards comparing their eigenvalues. Note, that this
    function only works for defect systems where the folder structure has
    been created with asr.setup.defects!
    """
    import numpy as np
    from asr.core import read_json

    try:
        p = Path('.')
        # sc = str(p.absolute()).split('/')[-2].split('_')[1].split('.')[0]
        pristinelist = list(p.glob(f'./../../defects.pristine_sc*/'))
        pris_folder = pristinelist[0]
        _, calc_pris = restart(pris_folder / 'gs.gpw', txt=None)
        res_pris = read_json(pris_folder / 'results-asr.gs.json')
        # res_def = read_json('results-asr.gs.json')
    except FileNotFoundError:
        print('ERROR: does not find pristine gs, pristine results, or defect'
              ' results. Did you run setup.defects and calculate the ground'
              ' state for defect and pristine system?')

    # for 2D systems, get pristine vacuum level
    if np.sum(calc.get_atoms().get_pbc()) == 2:
        eref_pris = res_pris['evac']
    # for 3D systems, set reference to zero (vacuum level doesn't make sense)
    else:
        eref_pris = 0

    # evaluate pristine VBM and CBM
    vbm = res_pris['vbm'] - eref_pris
    cbm = res_pris['cbm'] - eref_pris

    # reference pristine eigenvalues, get defect eigenvalues, align lowest
    # lying state of defect and pristine system
    es_pris = calc_pris.get_eigenvalues() - eref_pris
    es_def = calc.get_eigenvalues()
    dif = es_def[0] - es_pris[0]
    es_def = es_def - dif
    ef_def = calc.get_fermi_level() - dif

    # evaluate whether there are states above or below the fermi level
    # and within the bandgap
    above = False
    below = False
    for state in es_def:
        if state < cbm and state > vbm and state > ef_def:
            above = True
        elif state < cbm and state > vbm and state < ef_def:
            below = True
    above_below = (above, below)

    # evaluate states within the gap
    statelist = []
    [statelist.append(i) for i, state in enumerate(es_def) if (
        state < cbm and state > vbm)]

    return statelist, above_below, dif


def return_erange_states(calc, erange):
    """Evaluate states within a certain energy range wrt. the Fermi level."""
    es = calc.get_eigenvalues()
    ef = calc.get_fermi_level()

    statelist = []
    [statelist.append(i) for i, state in enumerate(es) if (
        state >= (ef + erange[0]) and state <= (ef + erange[1]))]

    return statelist


if __name__ == '__main__':
    main.cli()
