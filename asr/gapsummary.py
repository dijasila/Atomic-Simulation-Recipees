def webpanel2(row, key_descriptions):
    from typing import List
    from asr.utils.custom import val2str
    exclude = {}
    xcends = ['', '_gllbsc', '_hse', '_gw']
    xcs = ['PBE', 'GLLBSC', 'HSE', 'GW']
    if any(row.get(gap, 0) > 0 for gap in ['gap{}'.format(e) for e in xcends]):

        def methodtable(prefixes: List[str],
                        xcends: List[str] = xcends,
                        xcs: List[str] = xcs,
                        row=row) -> List[List[str]]:
            exclude.update(
                [prefix + end for end in xcends for prefix in prefixes])
            table = []
            for xc, end in zip(xcs, xcends):
                r = [val2str(row, prefix + end) for prefix in prefixes]
                if not any(r):
                    continue
                table.append([xc] + r)
            return table

        gaptable = dict(
            header=['Method', 'Band gap (eV)', 'Direct band gap (eV)'],
            type='table',
            rows=methodtable(('gap', 'dir_gap')))
        edgetable = dict(
            header=['Method', 'VBM vs vac. (eV)', 'CBM vs vac. (eV)'],
            type='table',
            rows=methodtable(prefixes=('vbm', 'cbm')))

        panel = ('Band gaps and -edges (all methods)', [[
            gaptable,
        ], [
            edgetable,
        ]])
    else:
        panel = ()
    things = ()
    return panel, things


group = 'postprocessing'
