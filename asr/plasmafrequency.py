from asr.core import command, option

@command('asr.plasmafrequency',
         creates=['es_plasma.gpw'])
@option('--kptdensity', help='k-point density')
def calculate(kptdensity=20):
    """Calculate excited states for polarizability calculation"""
    from gpaw import GPAW

    def get_kpts_size(atoms, density):
        """trying to get a reasonable monkhorst size which hits high
        symmetry points
        """
        from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
        size, offset = k2so(atoms=atoms, density=density)
        size[2] = 1
        for i in range(2):
            if size[i] % 6 != 0:
                size[i] = 6 * (size[i] // 6 + 1)
        kpts = {'size': size, 'gamma': True}
        return kpts

    calc_old = GPAW('gs.gpw', txt=None)
    kpts = get_kpts_size(atoms=calc_old.atoms, density=kptdensity)
    nval = calc_old.wfs.nvalence
    calc = GPAW('gs.gpw', fixdensity=True, kpts=kpts,
                nbands=2 * nval, txt='gsplasma.txt')
    calc.get_potential_energy()
    calc.write('es_plasma.gpw', 'all')


def webpanel(row, key_descriptions):
    from asr.browser import table

    plasmatable = table(row, 'Property', [
        'plasmafrequency_x', 'plasmafrequency_y'], key_descriptions)

    panel = {'title': 'Polarizability (RPA)',
             'columns': [[], [plasmatable]]}
    return [panel]


@command('asr.plasmafrequency',
         requires=['es_plasma.gpw'],
         webpanel=webpanel,
         dependencies=['asr.plasmafrequency@calculate'])
@option('--tetra', is_flag=True,
        help='Use tetrahedron integration')
def main(tetra=True):
    """Calculate polarizability"""
    from gpaw.response.df import DielectricFunction
    from ase.io import read
    import numpy as np
    from ase.units import Hartree, Bohr

    atoms = read('structure.json')
    nd = sum(atoms.pbc)
    if not nd == 2:
        raise AssertionError('Plasmafrequency recipe only implemented for 2D')
    
    if tetra:
        kwargs = {'truncation': '2D',
                  'eta': 0.05,
                  'domega0': 0.2,
                  'integrationmode': 'tetrahedron integration',
                  'ecut': 1,
                  'pbc': [True, True, False]}
    else:
        kwargs = {'truncation': '2D',
                  'eta': 0.05,
                  'domega0': 0.2,
                  'ecut': 1}

    df = DielectricFunction('es_plasma.gpw', **kwargs)
    df.get_polarizability(q_c=[0, 0, 0], direction='x',
                          pbc=[True, True, False],
                          filename=None)
    plasmafreq_vv = df.chi0.plasmafreq_vv.real
    data = {'plasmafreq_vv': plasmafreq_vv,
            '__key_descriptions__': {'plasmafreq_vv':
                                     'Plasma frequency tensor [Hartree]'}}

    if nd == 2:
        wp2_v = np.linalg.eigvalsh(plasmafreq_vv[:2, :2])
        L = atoms.cell[2, 2] / Bohr
        plasmafreq_v = (np.sqrt(wp2_v * L / 2) *
                        Hartree * Bohr**0.5)
        data['plasmafrequency_x'] = plasmafreq_v[0].real
        data['plasmafrequency_y'] = plasmafreq_v[1].real

        data['__key_descriptions__']['plasmafrequency_x'] = \
            'KVP: 2D Plasma frequency, x-direction [eV/Ang^0.5]'
        data['__key_descriptions__']['plasmafrequency_y'] = \
            'KVP: 2D Plasma frequency, y-direction [eV/Ang^0.5]'

    return data


if __name__ == '__main__':
    main.cli()
