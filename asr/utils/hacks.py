def get_parameter_description(recipename, parameters):
    from asr.database.browser import format_parameter_description
    desc = format_parameter_description(
        recipename,
        parameters,
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))
    return desc
