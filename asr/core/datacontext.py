from dataclasses import dataclass
from ase.utils import lazymethod, lazyproperty


class RecordNotFound(LookupError):
    pass


class DataContext:
    # The context object provides information which is
    # relevant for web panels but is not on the result object itself.
    # This includes access to input parameters, record and dependencies,
    # and so on.

    def __init__(self, row, record):
        self.row = row
        self.record = record

    @lazyproperty
    def descriptions(self):
        from asr.database.app import create_key_descriptions
        return create_key_descriptions()

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

    def gs_parameters(self):
        # XXX Some records do not depend on main, only calculate.
        # E.g. projected_bandstructure
        return self.get_record('asr.gs:calculate').parameters

    def get_record(self, name):
        if ':' not in name:
            name += ':main'  # XXX fixme

        records = self._dependencies()
        matches = [record for record in records if record.name == name]

        if self.name == name:
            matches.append(self.record)

        if len(matches) == 0:
            raise RecordNotFound(name)

        if len(matches) > 1:
            raise RuntimeError(f'Expected one {name} record, '
                               f'found: {matches}')

        return matches[0]

    def ground_state(self):
        return self.get_record('asr.gs')

    def magstate(self):
        return self.get_record('asr.magstate')

    def magnetic_anisotropy(self):
        return self.get_record('asr.magnetic_anisotropy')

    def bandstructure(self):
        return self.get_record('asr.bandstructure')

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
        if ndim == 2:
            assert all(atoms.pbc == [True, True, False])
        elif ndim == 1:
            assert all(atoms.pbc == [False, False, True])

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

        # XXX The GS recipe only sets 'evac' if we have exactly
        # two dimensions.  We should fix this.  Only in 3D do we
        # not have a vacuum level.
        if self.ndim != 2:
            return EnergyReference._efermi(gs['efermi'])
        else:
            return EnergyReference._evac(gs['evac'])

    def parameter_description(self, recipename, exclude=tuple()):
        from asr.database.browser import format_parameter_description
        parameters = self.get_record(recipename).parameters
        desc = format_parameter_description(
            recipename,
            parameters,
            exclude_keys=set(exclude))
        return desc

    def parameter_description_picky(self, recipename):
        boring = {'txt', 'fixdensity', 'verbose', 'symmetry',
                  'idiotproof', 'maxiter', 'hund', 'random',
                  'experimental', 'basis', 'setups'}
        return self.parameter_description(recipename, exclude=boring)


@dataclass
class EnergyReference:
    key: str
    value: float
    prose_name: str
    abbreviation: str

    @classmethod
    def _evac(cls, value):
        assert value is not None
        return cls('evac', value, 'vacuum level', 'vac')

    @classmethod
    def _efermi(cls, value):
        assert value is not None
        return cls('efermi', value, 'Fermi level', 'F')

    def mpl_plotlabel(self):
        return rf'$E - E_\mathrm{{{self.abbreviation}}}$ [eV]'

    def html_plotlabel(self):
        return rf'<i>E</i> − <i>E</i><sub>{self.abbreviation}</sub> [eV]'
