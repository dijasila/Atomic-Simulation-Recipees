def get_info():
    from pathlib import Path
    import json

    # Load parameters from params.json
    if Path('info.json').is_file():
        info = json.load(open('info.json', 'r'))
    else:
        info = {}

    return info


def get_parameters():
    from pathlib import Path
    import json
    if Path('params.json').is_file():
        params = json.load(open('params.json', 'r'))
    else:
        params = {}

    return params


def get_state():
    info = get_info()
    state = info.get('state', None)

    return state
