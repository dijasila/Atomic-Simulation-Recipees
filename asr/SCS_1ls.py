from asr.core import command, option

@command('asr.SCS_1ls')
@option('--structure', type = str)
@option('--kpoints', type = int)
@option('--eta', type = float)
def main(structure: str = None, kpoints: int = 18, eta: float = 0.01):
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

    if structure == "l1.json":
        gs_name = "l1.gpw"
    elif structure == "l2.json":
        gs_name = "l2.gpw"
    else:
        raise AssertionError('Only use this recipe for the single layers in the SCS calc!')


    # Loading the structure
    atoms = read(structure)
    # Setup the calculator 
    calc = GPAW(mode='lcao',
            xc='PBE',
            basis='dzp',
            kpts=(kpoints, kpoints, 1),
            occupations=FermiDirac(eta),
            txt='asr.bilayers_scs_layer1.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(gs_name, 'all')


if __name__ == "__main__":
    main.cli()
