from asr.utils import command, option


@command('asr.setup.magnetize',
         save_results_file=False)
@option('--state', type=str, default='all', help='Choices: all, nm, fm, afm')
@option('--name', default='unrelaxed.json',
        help='Atomic structure')
@option('--copy-params', default=False, is_flag=True,
        help='Also copy params.json from this dir (if exists).')
def main(state, name, copy_params):
    """Set up magnetic moments of atomic structure."""
    from pathlib import Path
    from ase.io import read, write
    from ase.parallel import world
    from asr.utils import magnetic_atoms
    import numpy as np
    known_states = ['nm', 'fm', 'afm']

    if state == 'all':
        states = known_states
    else:
        states = [state]

    for state in states:
        msg = f'{state} is not a known state!'
        assert state in known_states, msg

    # Non-magnetic:
    if 'nm' in states:
        atoms = read(name)
        atoms.set_initial_magnetic_moments(None)
        assert not Path('nm').is_dir(), 'nm/ already exists!'
        if world.rank == 0:
            Path('nm').mkdir()
            write('nm/unrelaxed.json', atoms)
            if copy_params:
                p = Path('params.json')
                if p.is_file:
                    Path('nm/params.json').write_text(p.read_text())

    # Ferro-magnetic:
    if 'fm' in states:
        atoms = read(name)
        atoms.set_initial_magnetic_moments([1] * len(atoms))
        assert not Path('fm').is_dir(), 'fm/ already exists!'
        if world.rank == 0:
            Path('fm').mkdir()
            write('fm/unrelaxed.json', atoms)
            if copy_params:
                p = Path('params.json')
                if p.is_file:
                    Path('fm/params.json').write_text(p.read_text())

    # Antiferro-magnetic:
    if 'afm' in states:
        atoms = read(name)
        magnetic = magnetic_atoms(atoms)
        nmag = sum(magnetic)
        if nmag == 2:
            magmoms = np.zeros(len(atoms))
            a1, a2 = np.where(magnetic)[0]
            magmoms[a1] = 1.0
            magmoms[a2] = -1.0
            atoms.set_initial_magnetic_moments(magmoms)
            assert not Path('afm').is_dir(), 'afm/ already exists!'
            if world.rank == 0:
                Path('afm').mkdir()
                write('afm/unrelaxed.json', atoms)
                if copy_params:
                    p = Path('params.json')
                    if p.is_file:
                        Path('afm/params.json').write_text(p.read_text())
        else:
            print('Warning: Did not produce afm state. '
                  f'The number of magnetic atoms is {nmag}. '
                  'At the moment, I only know how to do AFM '
                  'state for 2 magnetic atoms.')


group = 'setup'


if __name__ == '__main__':
    main()
