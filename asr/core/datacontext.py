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
        matches = [record for record in self._dependencies()
                   if record.name == name]

        if self.name == name:
            matches.append(self.record)

        if len(matches) != 1:
            raise RuntimeError(f'Expected one {name} record, '
                               f'found: {matches}')

        return matches[0]

    def ground_state(self):
        return self._find_dependency('asr.gs:main')

    def magstate(self):
        return self._find_dependency('asr.magstate:main')

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
