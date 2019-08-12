from asr.utils import command, option


def get_wavefunctions(atoms, name, params, density=6.0,
                      no_symmetries=False):
    from gpaw import GPAW
    from pathlib import Path

    params['kpts'] = {'density': density,
                      'gamma': True,
                      'even': True}
    if no_symmetries:
        params['symmetry'] = {'point_group': False,
                              'time_reversal': False}
    else:
        params['symmetry'] = {'point_group': True,
                              'time_reversal': True}
    params['convergence']['eigenstates'] = 1e-11
    tmp = Path(name).with_suffix('').name
    atoms.calc = GPAW(txt=tmp + '.txt', **params)
    atoms.get_potential_energy()
    atoms.calc.write(name, 'all')
    return atoms.calc


@command('asr.borncharges')
@option('--displacement', default=0.01, help='Atomic displacement (Å)')
@option('--kptdensity', default=6.0)
@option('--folder', default='data-borncharges')
def main(displacement, kptdensity, folder):
    """Calculate Born charges"""
    import json
    from os.path import exists, isfile
    from os import remove, makedirs
    from glob import glob

    import numpy as np
    from gpaw import GPAW
    from gpaw.mpi import world
    from asr.utils.berryphase import get_polarization_phase

    from ase.parallel import paropen
    from ase.units import Bohr
    from ase.io import jsonio

    from asr.collect import chdir

    if folder is None:
        folder = 'data-borncharges'

    if world.rank == 0:
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    world.barrier()

    with chdir(folder):
        calc = GPAW('../gs.gpw', txt=None)
        params = calc.parameters
        atoms = calc.atoms
        cell_cv = atoms.get_cell() / Bohr
        vol = abs(np.linalg.det(cell_cv))
        sym_a = atoms.get_chemical_symbols()

        pos_av = atoms.get_positions().copy()
        atoms.set_positions(pos_av)
        Z_avv = []
        P_asvv = []

        if world.rank == 0:
            print('Atomnum Atom Direction Displacement')
        for a in range(len(atoms)):
            phase_scv = np.zeros((2, 3, 3), float)
            for v in range(3):
                for s, sign in enumerate([-1, 1]):
                    if world.rank == 0:
                        print(sym_a[a], a, v, s)
                    # Update atomic positions
                    atoms.positions = pos_av
                    atoms.positions[a, v] = pos_av[a, v] + sign * displacement
                    prefix = 'born-{}-{}{}{}'.format(displacement, a,
                                                     'xyz'[v],
                                                     ' +-'[sign])
                    name = prefix + '.gpw'
                    berryname = prefix + '-berryphases.json'
                    if not exists(name) and not exists(berryname):
                        calc = get_wavefunctions(atoms, name, params,
                                                 density=kptdensity)
                    try:
                        phase_c = get_polarization_phase(name)
                    except ValueError:
                        calc = get_wavefunctions(atoms, name, params,
                                                 density=kptdensity)
                        phase_c = get_polarization_phase(name)

                    phase_scv[s, :, v] = phase_c

                    if exists(berryname):  # Calculation done?
                        if world.rank == 0:
                            # Remove gpw file
                            if isfile(name):
                                remove(name)

            dphase_cv = (phase_scv[1] - phase_scv[0])
            mod_cv = np.round(dphase_cv / (2 * np.pi)) * 2 * np.pi
            dphase_cv -= mod_cv
            phase_scv[1] -= mod_cv
            dP_vv = (-np.dot(dphase_cv.T, cell_cv).T /
                     (2 * np.pi * vol))

            P_svv = (-np.dot(cell_cv.T, phase_scv).transpose(1, 0, 2) /
                     (2 * np.pi * vol))
            Z_vv = dP_vv * vol / (2 * displacement / Bohr)
            P_asvv.append(P_svv)
            Z_avv.append(Z_vv)

        data = {'Z_avv': Z_avv, 'sym_a': sym_a,
                'P_asvv': P_asvv}

        filename = 'borncharges-{}.json'.format(displacement)

        with paropen(filename, 'w') as fd:
            json.dump(jsonio.encode(data), fd)

        world.barrier()
        if world.rank == 0:
            files = glob('born-*.gpw')
            for f in files:
                if isfile(f):
                    remove(f)


def polvsatom(row, *filenames):
    import numpy as np
    if 'borndata' not in row.data:
        return

    from matplotlib import pyplot as plt
    borndata = row.data['borndata']
    deltas = borndata[0]
    P_davv = borndata[1]

    for a, P_dvv in enumerate(P_davv.transpose(1, 0, 2, 3)):
        fname = 'polvsatom{}.png'.format(a)
        for fname2 in filenames:
            if fname in fname2:
                break
        else:
            continue

        Pm_vv = np.mean(P_dvv, axis=0)
        P_dvv -= Pm_vv
        plt.plot(deltas, P_dvv[:, 0, 0], '-o', label='xx')
        plt.plot(deltas, P_dvv[:, 1, 1], '-o', label='yy')
        plt.plot(deltas, P_dvv[:, 2, 2], '-o', label='zz')
        plt.xlabel('Displacement (Å)')
        plt.ylabel('Pol')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname2)
        plt.close()


def webpanel(row, key_descriptions):
    from asr.utils.custom import fig
    polfilenames = []
    if 'Z_avv' in row.data:
        def matrixtable(M, digits=2):
            table = M.tolist()
            shape = M.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = table[i][j]
                    table[i][j] = '{:.{}f}'.format(value, digits)
            return table

        columns = [[], []]
        for a, Z_vv in enumerate(row.data.Z_avv):
            Zdata = matrixtable(Z_vv)

            Ztable = dict(
                header=[str(a), row.symbols[a], ''],
                type='table',
                rows=Zdata)

            columns[0].append(Ztable)
            polname = 'polvsatom{}.png'.format(a)
            columns[1].append(fig(polname))
            polfilenames.append(polname)
        panel = ('Born charges', columns)
    else:
        panel = []
    things = ()
    return panel, things


def collect_data(atoms):
    import json
    import os.path as op
    import numpy as np
    from ase.io import jsonio

    kvp = {}
    data = {}
    key_descriptions = {}
    delta = 0.01
    P_davv = []
    fname = 'data-borncharges/borncharges-{}.json'.format(delta)
    if not op.isfile(fname):
        return {}, {}, {}

    with open(fname) as fd:
        dct = jsonio.decode(json.load(fd))

    P_davv.append(dct['P_asvv'][:, 0])
    P_davv.append(dct['P_asvv'][:, 1])
    data['Z_avv'] = -dct['Z_avv']

    P_davv = np.array(P_davv)
    data['borndata'] = [[-0.01, 0.01], P_davv]

    return kvp, key_descriptions, data


def print_results(filename='data-borncharges/borncharges-0.01.json'):
    import numpy as np
    import json
    from ase.io import jsonio
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    import os.path as op
    if not op.isfile(filename):
        return

    with open(filename) as fd:
        dct = jsonio.decode(json.load(fd))
    title = """
    BORNCHARGES
    ===========
    """
    print(title)
    print(-dct['Z_avv'])


group = 'property'
dependencies = ['asr.structureinfo', 'asr.gs']
resources = '24:10h'


if __name__ == '__main__':
    main()
