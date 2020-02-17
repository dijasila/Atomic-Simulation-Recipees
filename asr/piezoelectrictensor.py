def webpanel(row, key_descriptions):
    for i in range(1, 4):
        for j in range(1, 7):
            key = 'e0_{}{}'.format(i, j)
            name = 'Clamped piezoelectric tensor'
            description = ('{} ({}{})'.format(name, i, j),
                           '{} ({}{}-component)'.format(name, i, j),
                           '`\\text{Ang}^{-1}`')
            key_descriptions[key] = description

            key = 'e_{}{}'.format(i, j)
            name = 'Piezoelectric tensor'
            description = ('{} ({}{})'.format(name, i, j),
                           '{} ({}{}-component)'.format(name, i, j),
                           '`\\text{Ang}^{-1}`')
            key_descriptions[key] = description

    if 'e_vvv' in row.data:
        def matrixtable(M, digits=2):
            table = M.tolist()
            shape = M.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = table[i][j]
                    table[i][j] = '{:.{}f}'.format(value, digits)
            return table

        e_ij = row.data.e_vvv[:, [0, 1, 2, 1, 0, 0],
                              [0, 1, 2, 2, 2, 1]]
        e0_ij = row.data.e0_vvv[:, [0, 1, 2, 1, 0, 0],
                                [0, 1, 2, 2, 2, 1]]

        etable = dict(
            header=['Piezoelectric tensor', '', ''],
            type='table',
            rows=matrixtable(e_ij))

        e0table = dict(
            header=['Clamped piezoelectric tensor', ''],
            type='table',
            rows=matrixtable(e0_ij))

        columns = [[etable, e0table], []]

        panel = [('Piezoelectric tensor', columns)]
    else:
        panel = ()
    things = ()
    return panel, things


group = 'property'
