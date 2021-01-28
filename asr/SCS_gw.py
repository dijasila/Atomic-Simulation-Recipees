from asr.core import command, option

@command('asr.bilayer_scs_gs')
@option('--kpoints', type = int)
@option('--bandpathpoints', type = float)
@option('--eta', type = float)
def main(kpoints: int = 18,
         bandpathpoints: int = 80,
         eta: float = 0.01):
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


    atoms = read('structure.json')


    # Define the projector
    c1_gs = GPAW('layer1_gs.gpw', txt = None)
    c1_bs = GPAW('layer1_bs.gpw', txt = None)
    c2_gs = GPAW('layer2_gs.gpw', txt = None)
    c2_bs = GPAW('layer2_bs.gpw', txt = None)
    
    shifts = json.load(open('shifts.json', 'r'))
    shift_v1 = shifts['shift_v1']
    shift_c1 = shifts['shift_c1']
    shift_v2 = shifts['shift_v2']
    shift_c2 = shifts['shift_c2']

    SO_gs = Scissors([(shift_v1, shift_c1, c1_gs), (shift_v2, shift_c2, c2_gs)])
    SO_bs = Scissors([(shift_v1, shift_c1, c1_bs), (shift_v2, shift_c2, c2_bs)])


    # Calculate the ground state with the scissors operator
    print("Runtime!")

    calc_gs = GPAW(mode='lcao',
                xc='PBE',
                basis='dzp',
                kpts=(kpoints, kpoints, 1),
                occupations=FermiDirac(eta),
                eigensolver=SO_gs, 
                txt='asr.scs_gs.txt')
    atoms.calc = calc_gs
    atoms.get_potential_energy()
    calc_gs.write("scs_gs.gpw", 'all')

    path = atoms.cell.bandpath(npoints=bandpathpoints, 
                               pbc=atoms.pbc, 
                               eps = 1e-2)

    calc_bs = GPAW("scs_gs.gpw",
                fixdensity=True,
                symmetry='off',
                eigensolver=SO_bs,
                kpts=path,
                txt='scs_bs.txt')
    calc_bs.get_potential_energy()
    calc_bs.write('scs_bs.gpw', 'all')


if __name__ == "__main__":
    main()
