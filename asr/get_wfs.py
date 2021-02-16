from ase.io import read, write
from asr.core import command, option, ASRResult, prepare_result
from gpaw import GPAW, restart
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc
import typing
import numpy as np
from pathlib import Path
from ase import Atoms


@prepare_result
class WaveFunctionResult(ASRResult):
    """Container for results of specific wavefunction for one spin
    channel."""
    state = int
    spin = int
    energy = float

    key_descriptions: typing.Dict[str, str] = dict(
        state='State index.',
        spin='Spin index (0 or 1).',
        energy='Energy of the state (ref. to the vacuum level in 2D) [eV].')

@prepare_result
class Result(ASRResult):
    """Container for asr.get_wfs results."""
    wfs: typing.List[WaveFunctionResult]
    above_below: typing.Tuple[bool, bool]

    key_descriptions: typing.Dict[str, str] = dict(
        wfs='List of WaveFunctionResult objects for all states.',
        above_below='States within the gap above and below EF? '
                    '(ONLY for defect systems).')


@command(module='asr.get_wfs',
         requires=['gs.gpw', 'structure.json'],
         resources='1:10m',
         returns=Result)
@option('--state', help='Specify state index that you want to '
        'write out. This option will not be used when '
        '"--get-gapstates" is used.', type=int)
@option('--get-gapstates/-dont-get-gapstates', help='Save all wfs '
        'for states present inside the gap (only use this option for '
        'defect systems, where your folder structure has been set up '
        'with asr.setup.defects).', is_flag=True)
def main(state: int = 0,
         get_gapstates: bool = False) -> Result:
    """Perform fixed density calculation and write out wavefunctions.

    This recipe reads in an existing gs.gpw file, runs a fixed density
    calculation on it, and writes out wavefunctions for a state in
    "wf.{bandindex}_{spin}.cube" cube files. This state can either be
    a particular state given by the bandindex with the option "--state",
    or (for defect systems) be all of the states in the gap which will
    be evaluated with the option "--get-gapstates". Note, that when
    "--get-gapstates" is given, "--state" will be ignored.
    """
    import numpy as np
    from gpaw import restart

    print('INFO: run fixed density calculation.')
    atoms, calc = restart('gs.gpw', txt='get_wfs.txt')
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})
    if get_gapstates:
        print('INFO: evaluate gapstates.')
        pass
    elif not get_gapstates:
        if np.sum(atoms.get_pbc()) == 2:
            eref = read_json('results-asr.gs.json')['evac']
        else:
            eref = 0
        states = [state]
        above_below = (None, None)

    energies_0 = []
    energies_1 = []
    wfs_results = []
    for state in states:
        wf = calc.get_pseudo_wave_function(band=band, spin=0)
        energy = calc.get_potential_energy() + eref
        energies_0.append(energy)
        fname = f'wf.{state}_0.cube'
        write(fname, atoms, data=wf)
        wfs_results.append(WaveFunctionResult.fromdata(
            state=state,
            spin=0,
            energy=energy))
        if calc.get_number_of_spins() == 2:
            wf = calc.get_pseudo_wave_function(band=band, spin=1)
            energy = calc.get_potential_energy() + eref
            energies_1.append(energy)
            fname = f'wf.{state}_1.cube'
            write(fname, atoms, data=wf)
            wfs_results.append(WaveFunctionResult.fromdata(
                state=state,
                spin=0,
                energy=energy))

    return Result.fromdata(
        wfs=wfs_results,
        above_below=above_below)
