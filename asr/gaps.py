# ##TODO min kpt dens?
import json
from asr.utils import command, option


@command('asr.gap')
@option('--gpwfilename', type=str, help='filename.gpw', default='gs.gpw')
def main(gpwfilename):
    from gpaw import GPAW
    from functools import partial
    from pathlib import Path
    from ase.parallel import paropen
    from asr.utils.gpw2eigs import gpw2eigs
    # inputs: gpw groundstate file, soc?, direct gap?
    if not Path(gpwfilename).is_file():
        raise ValueError('Groundstate file not present')
    calc = GPAW(gpwfilename, txt=None)
    ibzkpts = calc.get_ibz_k_points()

    for soc in [True, False]:
        evbm_ecbm_gap, skn_vbm, skn_cbm = get_gap_info(
            soc=soc, direct=False, calc=calc, gpw=gpwfilename)
        evbm_ecbm_direct_gap, direct_skn_vbm, direct_skn_cbm = get_gap_info(
            soc=soc, direct=True, calc=calc, gpw=gpwfilename)
        k_vbm, k_cbm = skn_vbm[1], skn_cbm[1]
        direct_k_vbm, direct_k_cbm = direct_skn_vbm[1], direct_skn_cbm[1]

        get_kc = partial(get_1bz_k, ibzkpts, calc)

        k_vbm_c = get_kc(k_vbm)
        k_cbm_c = get_kc(k_cbm)
        direct_k_vbm_c = get_kc(direct_k_vbm)
        direct_k_cbm_c = get_kc(direct_k_cbm)

        if soc:
            _, efermi = gpw2eigs(gpwfilename, soc=True,
                                 optimal_spin_direction=True)
        else:
            efermi = calc.get_fermi_level()

        data = {'gap': evbm_ecbm_gap[2],
                'vbm': evbm_ecbm_gap[0],
                'cbm': evbm_ecbm_gap[1],
                'gap_dir': evbm_ecbm_direct_gap[2],
                'vbm_dir': evbm_ecbm_direct_gap[0],
                'cbm_dir': evbm_ecbm_direct_gap[1],
                'k1_c': k_vbm_c,
                'k2_c': k_cbm_c,
                'k1_dir_c': direct_k_vbm_c,
                'k2_dir_c': direct_k_cbm_c,
                'skn1': skn_vbm,
                'skn2': skn_cbm,
                'skn1_dir': direct_skn_vbm,
                'skn2_dir': direct_skn_cbm,
                'efermi': efermi}

        with paropen('gap{}.json'.format('_soc' if soc else ''), 'w') as f:
            from ase.io.jsonio import MyEncoder
            f.write(MyEncoder(indent=4).encode(data))


def collect_data(atoms):
    from pathlib import Path

    data = {}
    kvp = {}
    key_descriptions = {}

    data_to_include = ['gap', 'vbm', 'cbm', 'gap_dir', 'vbm_dir', 'cbm_dir',
                       'efermi']
    descs = [('Bandgap', 'Bandgap', 'eV'),
             ('Valence Band Maximum', 'Maximum of valence band', 'eV'),
             ('Conduction Band Minimum', 'Minimum of conduction band', 'eV'),
             ('Direct Bandgap', 'Direct bandgap', 'eV'),
             ('Valence Band Maximum - Direct',
              'Valence Band Maximum - Direct', 'eV'),
             ('Conduction Band Minimum - Direct',
              'Conduction Band Minimum - Direct', 'eV'),
             ('Fermi Level', "Fermi's level", 'eV')]

    for soc in [True, False]:
        path = Path('gap{}.json'.format('_soc' if soc else ''))

        if not path.is_file():
            continue

        sdata = json.loads(path.read_text())

        keyname = 'soc' if soc else 'nosoc'
        data[keyname] = sdata

        def namemod(n):
            return n + '_soc' if soc else n

        includes = [namemod(n) for n in data_to_include]

        for k, inc in enumerate(includes):
            val = sdata[data_to_include[k]]
            if val is not None:
                kvp[inc] = val
                key_descriptions[inc] = descs[k]

    return kvp, key_descriptions, data


def webpanel(row, key_descriptions):
    from asr.utils.custom import table

    t = table(row, 'Postprocessing', [
              'gap', 'vbm', 'cbm', 'gap_dir', 'vbm_dir', 'cbm_dir', 'efermi'],
              key_descriptions)

    panel = ('Gap information', [[t]])

    return panel, None


def get_1bz_k(ibzkpts, calc, k_index):
    from gpaw.kpt_descriptor import to1bz
    k_c = ibzkpts[k_index] if k_index is not None else None
    if k_c is not None:
        k_c = to1bz(k_c[None], calc.wfs.gd.cell_cv)[0]
    return k_c


def get_gap_info(soc, direct, calc, gpw):
    from ase.dft.bandgap import bandgap
    from asr.utils.gpw2eigs import gpw2eigs
    # e1 is VBM, e2 is CBM
    if soc:
        e_km, efermi = gpw2eigs(gpw, soc=True, optimal_spin_direction=True)
        # km1 is VBM index tuple: (s, k, n), km2 is CBM index tuple: (s, k, n)
        gap, km1, km2 = bandgap(eigenvalues=e_km, efermi=efermi, direct=direct,
                                kpts=calc.get_ibz_k_points(), output=None)
        if km1[0] is not None:
            e1 = e_km[km1]
            e2 = e_km[km2]
        else:
            e1, e2 = None, None
        x = (e1, e2, gap), (0,) + tuple(km1), (0,) + tuple(km2)
    else:
        g, skn1, skn2 = bandgap(calc, direct=direct, output=None)
        if skn1[1] is not None:
            e1 = calc.get_eigenvalues(spin=skn1[0], kpt=skn1[1])[skn1[2]]
            e2 = calc.get_eigenvalues(spin=skn2[0], kpt=skn2[1])[skn2[2]]
        else:
            e1, e2 = None, None
        x = (e1, e2, g), skn1, skn2
    return x


def eigenvalues(calc):
    import numpy as np
    rs = range(calc.get_number_of_spins())
    rk = range(len(calc.get_ibz_k_points()))
    e = calc.get_eigenvalues
    return np.asarray([[e(spin=s, kpt=k) for k in rk] for s in rs])


group = 'postprocessing'
dependencies = ['asr.quickinfo', 'asr.gs']


if __name__ == '__main__':
    main()
