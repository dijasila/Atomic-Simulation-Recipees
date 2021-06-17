def gs_xcname_from_row(row):
    params = row.data.get_record('results-asr.gs@calculate.json').parameters
    if 'calculator' not in params:
        # What are the rules for when this piece of data exists?
        # Presumably the calculation used ASR defaults.
        return 'PBE'
    # If the parameters are present, but were not set, we are using
    # GPAW's default which is LDA.
    return params['calculator'].get('xc', 'LDA')


class RowInfo:
    """Utilities for retrieving data from database rows.

    This object provides methods that are used in many recipes.
    It exists to replace duplication across those recipes.  There are
    probably better designs."""

    def __init__(self, row):
        self.row = row

    @property
    def gsdata(self):
        return self.row.data['results-asr.gs@calculate.json']

    def gs_xcname(self):
        return gs_xcname_from_row(self.row)

    def have_evac(self):
        return self.get_evac() is not None

    def get_evac(self, default=None):
        return self.row.get('evac', default)

    def evac_or_efermi(self):
        # We should probably be getting this data from GS results, not row
        evac = self.get_evac()
        if evac is not None:
            return EnergyReference('evac', evac, 'vacuum level', 'vac')

        efermi = self.row.get('efermi')
        return EnergyReference('efermi', efermi, 'Fermi level', 'F')


class EnergyReference:
    def __init__(self, key, value, prose_name, abbreviation):
        self.key = key
        self.value = value
        self.prose_name = prose_name
        self.abbreviation = abbreviation

    def mpl_plotlabel(self):
        return rf'$E - E_\mathrm{{{self.abbreviation}}}$ [eV]'
