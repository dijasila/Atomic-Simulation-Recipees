from asr.utils import read_json
from pathlib import Path


def get_DFT_calculator():
    if not Path('params.json').is_file():
        name = 'GPAW'
    else:
        params = read_json('params.json')
        name = params.get('_calculator', 'GPAW')

    if name == 'GPAW':
        from gpaw import GPAW
        return GPAW
    elif name == 'EMT':
        from ase.calculators.emt import EMT
        import ase.io.ulm as ulm

        class ASREMT(EMT):
            def __init__(self, **kwargs):
                EMT.__init__(self)

            def write(self, filename):

                from ase.io.trajectory import write_atoms
                with ulm.open(filename, 'w') as w:
                    write_atoms(w.child('atoms'), self.atoms)
                    w.child('results').write(**self.results)
                    w.child('wave_functions').write(foo='bar')
                    w.child('occupations').write(fermilevel=42)
        return ASREMT
    else:
        raise NotImplementedError('Unknown DFT calculator')
