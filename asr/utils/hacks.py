def gs_xcname_from_row(row):
    return RowInfo(row).gs_xcname()


class RowInfo:
    def __init__(self, row):
        self.row = row

    @property
    def gsdata(self):
        return self.row.data['results-asr.gs@calculate.json']

    def gs_xcname(self):
        data = self.gsdata
        if not hasattr(data, 'metadata'):
            # Old (?) compatibility hack
            return 'PBE'
        params = data.metadata.params
        if 'calculator' not in params:
            # What are the rules for when this piece of data exists?
            # Presumably the calculation used ASR defaults.
            return 'PBE'
        # If the parameters are present, but were not set, we are using
        # GPAW's default which is LDA.
        return params['calculator'].get('xc', 'LDA')

    def evac_or_efermi(self):
        # We should probably be getting this data from GS results, not row
        evac = self.row.get('evac')
        if evac is not None:
            return EnergyReference('evac', evac, 'vacuum level')

        efermi = self.row.get('efermi')
        return EnergyReference('efermi', efermi, 'Fermi level')


class EnergyReference:
    def __init__(self, key, value, prose_name):
        self.key = key
        self.value = value
        self.prose_name = prose_name
