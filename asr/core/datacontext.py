from dataclasses import dataclass
from ase.utils import lazymethod
from asr.database.app import create_key_descriptions


class DataContext:
    descriptions = create_key_descriptions()
    # Can we find a more fitting name for this?
    #
    # Basically the context object provides information which is
    # relevant for web panels but is not on the result object itself.
    # So "DataContext" makes sense but sounds a little bit abstract.

    # We need to add whatever info from Records that's needed by web panels.
    # But not the records themselves -- they're not part of the database
    # and we would like it to be possible to generate the figures
    # from a database without additional info.
    def __init__(self, row, record):
        self.row = row
        self.record = record

    @property
    def parameters(self):
        return self.record.parameters

    @property
    def name(self):
        return self.record.name

    @property
    def result(self):
        return self.record.result

    @lazymethod
    def _dependencies(self):
        from asr.core.cache import get_cache
        cache = get_cache()
        # XXX Avoid depending directly on backend
        return list(cache.backend.recurse_dependencies(self.record))

    def _find_dependency(self, name):
        if ':' not in name:
            name += ':main'  # XXX fixme
        matches = [record for record in self._dependencies()
                   if record.name == name]

        if self.name == name:
            matches.append(self.record)

        if len(matches) != 1:
            raise RuntimeError(f'Expected one {name} record, '
                               f'found: {matches}')

        return matches[0]

    def ground_state(self):
        return self._find_dependency('asr.gs')

    def magstate(self):
        return self._find_dependency('asr.magstate')

    def magnetic_anisotropy(self):
        return self._find_dependency('asr.magnetic_anisotropy')

    def bandstructure(self):
        return self._find_dependency('asr.bandstructure')

    def gs_results(self):
        return self.ground_state().result

    @property
    def xcname(self):
        # XXX This is bound to GPAW's default XC functional.
        return self.parameters['calculator'].get('xc', 'LDA')

    @property
    def atoms(self):
        return self.parameters['atoms']

    @property
    def ndim(self):
        atoms = self.atoms
        ndim = sum(atoms.pbc)
        assert all(atoms.pbc[:ndim])
        return ndim

    @property
    def is_magnetic(self):
        return self.magstate().result.is_magnetic

    @property
    def spin_axis(self):
        record = self.magnetic_anisotropy()
        return record.result['spin_axis']

    # This pattern is used by almost all recipes that have spectra (?)
    def energy_reference(self):
        gs = self.gs_results()
        if self.ndim == 3:
            return EnergyReference._efermi(gs['efermi'])
        else:
            return EnergyReference._evac(gs['evac'])


@dataclass
class EnergyReference:
    key: str
    value: float
    prose_name: str
    abbreviation: str

    @classmethod
    def _evac(cls, value):
        return cls('evac', value, 'vacuum level', 'vac')

    @classmethod
    def _efermi(cls, value):
        return cls('efermi', value, 'Fermi level', 'F')

    def mpl_plotlabel(self):
        return rf'$E - E_\mathrm{{{self.abbreviation}}}$ [eV]'

    def html_plotlabel(self):
        return rf'<i>E</i> − <i>E</i><sub>{self.abbreviation}</sub> [eV]'
