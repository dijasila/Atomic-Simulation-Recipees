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
    from ase import Atoms
    from ase.io import read
    from ase.io.jsonio import read_json, write_json

    if structure == "l1.json":
        gs_name = "l1.gpw"
        layer_nr = 1
    elif structure == "l2.json":
        gs_name = "l2.gpw"
        layer_nr = 2
    else:
        raise AssertionError('Only use this recipe for the single layers in the SCS calc!')


    # Loading the structure
    dct = read_json(structure)
    atoms = Atoms.fromdict(dct)
    # Setup the calculator 
    calc = GPAW(mode='lcao',
            xc='PBE',
            basis='dzp',
            kpts=(kpoints, kpoints, 1),
            occupations=FermiDirac(eta),
            txt=f'asr.bilayers_scs_layer{layer_nr}.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(gs_name, 'all')


if __name__ == "__main__":
    main.cli()
