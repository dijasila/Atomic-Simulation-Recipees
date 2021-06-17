def gs_xcname_from_row(row):
    params = row.data.get_record('results-asr.gs@calculate.json').parameters
    if 'calculator' not in params:
        # What are the rules for when this piece of data exists?
        # Presumably the calculation used ASR defaults.
        return 'PBE'
    # If the parameters are present, but were not set, we are using
    # GPAW's default which is LDA.
    return params['calculator'].get('xc', 'LDA')


def get_parameter_description(recipename, parameters):
    from asr.database.browser import format_parameter_description
    desc = format_parameter_description(
        recipename,
        parameters,
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))
    return desc
