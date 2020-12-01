from asr.core import command, option

@command('asr.bilayer_scs_gs')
@option('--kpoints', type = int)
@option('--eta', type = float)
def main(kpoints: int = 6, eta: float = 0.01):
    '''
    This recipe uses the self-consistent scissors operator.
    OBS: Note that the SCS in defined in a development branch of GPAW 
    and has yet to be merged into the main version.
    '''
    import json
    from gpaw import GPAW
    from gpaw import GPAW, PW, FermiDirac
    from gpaw.lcao.scissors import Scissors
    from ase.io import read
    from ase.io.jsonio import read_json, write_json


    # Loading the structure
    atoms = read('structure.json')
    # Define the projector
    c1 = GPAW('layer1.gpw', txt = None)
    c2 = GPAW('layer2.gpw', txt = None)
    shifts = json.load(open('shifts.json', 'r'))

    shift_v1 = shifts['shift_v1']
    shift_c1 = shifts['shift_c1']
    shift_v2 = shifts['shift_v2']
    shift_c2 = shifts['shift_c2']
    SO = Scissors([(shift_v1, shift_c1, c1), (shift_v2, shift_c2, c2)])

    print("Runtime!")
    # Setup the calculator 
    calc = GPAW(mode='lcao',
            xc='PBE',
            basis='dzp',
            kpts=(kpoints, kpoints, 1),
            occupations=FermiDirac(eta),
            eigensolver=SO, 
            txt='asr.bilayers_scs_gs.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write("scs_gs.gpw", 'all')


if __name__ == "__main__":
    main()
