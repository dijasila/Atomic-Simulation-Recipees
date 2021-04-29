def gs_xcname_from_row(row):
    # XXX Huge trainwreck
    record = row.data.get_record('results-asr.gs@calculate.json')
    return record.parameters['calculator'].get('xc', 'LDA')
