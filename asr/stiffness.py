from asr.core import command, option

tests = [{'cli': ['ase build -x diamond Si structure.json',
                  'asr run "setup.strains --kptdensity 2.0"',
                  'asr run "setup.params asr.relax:calculator '
                  '''{'mode':{'ecut':200},'kpts':(1,1,1),...}" strains*/''',
                  'asr run relax strains*/',
                  'asr run database.material_fingerprint strains*/',
                  'asr run stiffness']}]


def webpanel(row, key_descriptions):
    import numpy as np

    def matrixtable(M, digits=2, unit='', skiprow=0, skipcolumn=0):
        table = M.tolist()
        shape = M.shape

        for i in range(skiprow, shape[0]):
            for j in range(skipcolumn, shape[1]):
                value = table[i][j]
                table[i][j] = '{:.{}f}{}'.format(value, digits, unit)
        return table
    stiffnessdata = row.data['results-asr.stiffness.json']
    c_ij = stiffnessdata['stiffness_tensor']
    eigs = stiffnessdata['eigenvalues']

    c_ij = np.zeros((4, 4))
    c_ij[1:, 1:] = stiffnessdata['stiffness_tensor']
    rows = matrixtable(c_ij, unit='',
                       skiprow=1,
                       skipcolumn=1)
    rows[0] = ['C<sub>ij</sub> (N/m)', 'xx', 'yy', 'xy']
    rows[1][0] = 'xx'
    rows[2][0] = 'yy'
    rows[3][0] = 'xy'
    for ir, tmprow in enumerate(rows):
        for ic, item in enumerate(tmprow):
            if ir == 0 or ic == 0:
                rows[ir][ic] = '<b>' + rows[ir][ic] + '</b>'

    ctable = dict(
        type='table',
        rows=rows)

    eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']] +
               [[f'Eigenvalue {ie}', f'{eig:.2f} N/m']
                for ie, eig in enumerate(sorted(eigs,
                                                key=lambda x: x.real))])
    eigtable = dict(
        type='table',
        rows=eigrows)

    panel = {'title': 'Stiffness tensor',
             'columns': [[ctable], [eigtable]],
             'sort': 2}

    dynstab = ['low', 'high'][int(eigs.min() > 0)]
    high = 'Min. Stiffness eig. > 0'
    low = 'Min. Stiffness eig. < 0'
    row = ['Dynamical (stiffness)',
           '<a href="#" data-toggle="tooltip" data-html="true" ' +
           'title="LOW: {}&#13;HIGH: {}">{}</a>'.format(
               low, high, dynstab.upper())]

    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Stability', 'Category'],
                             'rows': [row],
                             }]],
               'sort': 3}

    return [panel, summary]


@command(module='asr.stiffness',
         webpanel=webpanel,
         tests=tests)
@option('--strain-percent', help='Magnitude of applied strain')
def main(strain_percent=1.0):
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)
    from ase.io import read
    from ase.units import J
    from asr.core import read_json, chdir
    from asr.database.material_fingerprint import main as computemf
    import numpy as np
    
    atoms = read('structure.json')
    ij = get_relevant_strains(atoms.pbc)

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    links = {}
    stiffness = np.zeros((6, 6), float)
    for i, j in ij:
        dstress = np.zeros((6,), float)
        for sign in [-1, 1]:
            folder = get_strained_folder_name(sign * strain_percent, i, j)
            structurefile = folder / 'structure.json'
            if not computemf.done:
                with chdir(folder):
                    computemf()
            mf = read_json(folder / ('results-asr.database.'
                                     'material_fingerprint.json'))
            links[str(folder)] = mf['uid']
            structure = read(str(structurefile))
            # The structure already has the stress if it was
            # calculated
            stress = structure.get_stress(voigt=True)
            dstress += stress * sign
        stiffness[:, ij_to_voigt[i][j]] = dstress / (strain_percent * 0.02)

    stiffness = np.array(stiffness, float)
    # We work with Mandel notation which is conventional and convenient
    stiffness[3:, :] *= 2**0.5
    stiffness[:, 3:] *= 2**0.5

    # Convert the stiffness tensor from [eV/Ang^3] -> [J/m^3]=[N/m^2]
    stiffness *= 10**30 / J

    # Now do some post processing
    data = {'__key_descriptions__': {}}
    kd = data['__key_descriptions__']
    nd = np.sum(atoms.pbc)
    if nd == 2:
        cell = atoms.get_cell()
        # We have to normalize with the supercell size
        z = cell[2, 2]
        stiffness = stiffness[[0, 1, 5], :][:, [0, 1, 5]] * z * 1e-10
        from ase.units import kg
        from ase.units import m as meter
        area = atoms.get_volume() / cell[2, 2]
        mass = sum(atoms.get_masses())
        area_density = (mass / kg) / (area / meter**2)
        # speed of sound in m/s
        speed_x = np.sqrt(stiffness[0, 0] / area_density)
        speed_y = np.sqrt(stiffness[1, 1] / area_density)
        data['speed_of_sound_x'] = speed_x
        data['speed_of_sound_y'] = speed_y
        data['c_11'] = stiffness[0, 0]
        data['c_22'] = stiffness[1, 1]
        data['c_33'] = stiffness[2, 2]
        data['c_23'] = stiffness[1, 2]
        data['c_13'] = stiffness[0, 2]
        data['c_12'] = stiffness[0, 1]
        kd['c_11'] = 'KVP: Stiffness tensor: 11-component [N/m]'
        kd['c_22'] = 'KVP: Stiffness tensor: 22-component [N/m]'
        kd['c_33'] = 'KVP: Stiffness tensor: 33-component [N/m]'
        kd['c_23'] = 'KVP: Stiffness tensor: 23-component [N/m]'
        kd['c_13'] = 'KVP: Stiffness tensor: 13-component [N/m]'
        kd['c_12'] = 'KVP: Stiffness tensor: 12-component [N/m]'
        kd['speed_of_sound_x'] = 'KVP: Speed of sound in x direction [m/s]'
        kd['speed_of_sound_y'] = 'KVP: Speed of sound in y direction [m/s]'
        kd['stiffness_tensor'] = 'Stiffness tensor [N/m]'
    elif nd == 1:
        area = atoms.get_volume() / cell[2, 2]
        stiffness = stiffness[5, 5] * area * 1e-20
        kd['stiffness_tensor'] = 'Stiffness tensor [N]'
    else:
        kd['stiffness_tensor'] = 'Stiffness tensor [N/m^2]'

    data['__links__'] = links
    data['stiffness_tensor'] = stiffness

    eigs = np.linalg.eigvals(stiffness)
    data['eigenvalues'] = eigs
    return data


if __name__ == '__main__':
    main.cli()
