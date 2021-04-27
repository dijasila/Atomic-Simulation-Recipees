def gs_xcname_from_row(row):
    # XXX Huge trainwreck
    params = row.data['results-asr.gs@calculate.json'].metadata.params
    return params['calculator'].get('xc', 'LDA')
