def gs_xcname_from_row(row):
    data = row.data['results-asr.gs@calculate.json']
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
