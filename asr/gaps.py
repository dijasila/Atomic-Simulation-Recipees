# ##TODO min kpt dens?
from asr.utils import click, update_defaults, get_start_parameters
params = get_start_parameters()
defaults = {}


@click.command()
@update_defaults('asr.gs', defaults)
@click.option('--gpwfilename', type=str, help='filename.gpw', default='gs.gpw')
def main(gpwfilename):
    import numpy as np
    from gpaw import GPAW
    from functools import partial
    from pathlib import Path
    from ase.parallel import paropen
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

            with paropen('gap{}.npz'.format('_soc' if soc else ''), 'wb') as f:
                np.savez(f, **data)
            # Path('gap{}.json'.format('_soc' if soc else '')).write_text(
            #    json.dumps(data))


def collect_data(atoms):
    import numpy as np
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
        fname = 'gap{}.npz'.format('_soc' if soc else '')

        if not Path(fname).is_file():
            continue

        sdata = np.load(fname)

        keyname = 'soc' if soc else 'nosoc'
        data[keyname] = sdata

        def namemod(n):
            return n + '_soc' if soc else n

        includes = [namemod(n) for n in data_to_include]

        for k, inc in enumerate(includes):
            kvp[inc] = sdata[data_to_include[k]]
            key_descriptions[inc] = descs[k]

    return kvp, key_descriptions, data


def webpanel(row, key_descriptions):
    from asr.custom import table

    t = table(row, 'Postprocessing', [
              'gap', 'vbm', 'cbm', 'gap_dir', 'vbm_dir', 'cbm_dir', 'efermi'],
              key_descriptions)

    panel = ('Gap information', [t])

    return panel, None


def get_1bz_k(ibzkpts, calc, k_index):
    from gpaw.kpt_descriptor import to1bz
    k_c = ibzkpts[k_index] if k_index is not None else None
    if k_c is not None:
        k_c = to1bz(k_c[None], calc.wfs.gd.cell_cv)[0]
    return k_c


def get_gap_info(soc, direct, calc, gpw):
    from ase.dft.bandgap import bandgap
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


def gpw2eigs(gpw, soc=True, bands=None, return_spin=False,
             optimal_spin_direction=False):
    from gpaw import GPAW
    from gpaw.spinorbit import get_spinorbit_eigenvalues
    from gpaw import mpi
    from ase.parallel import broadcast
    import numpy as np
    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    dct = None
    if mpi.world.rank in ranks:
        theta = 0
        phi = 0
        if optimal_spin_direction:
            theta, phi = get_spin_direction()
        calc = GPAW(gpw, txt=None, communicator=comm)
        if bands is None:
            n2 = calc.todict().get("convergence", {}).get("bands")
            bands = slice(0, n2)
        if isinstance(bands, slice):
            bands = range(calc.get_number_of_bands())[bands]
        eps_nosoc_skn = eigenvalues(calc)[..., bands]
        efermi_nosoc = calc.get_fermi_level()
        eps_mk, s_kvm = get_spinorbit_eigenvalues(calc, bands=bands,
                                                  theta=theta,
                                                  phi=phi,
                                                  return_spin=True)
        eps_km = eps_mk.T
        efermi = fermi_level(calc, eps_km[np.newaxis],
                             nelectrons=2 * calc.get_number_of_electrons())
        dct = {'eps_nosoc_skn': eps_nosoc_skn,
               'eps_km': eps_km,
               'efermi_nosoc': efermi_nosoc,
               'efermi': efermi,
               's_kvm': s_kvm}
    dct = broadcast(dct, root=0, comm=mpi.world)
    if soc is None:
        return dct
    elif soc:
        out = (dct['eps_km'], dct['efermi'], dct['s_kvm'])
        if not return_spin:
            out = out[:2]
        return out
    else:
        return dct['eps_nosoc_skn'], dct['efermi_nosoc']


def fermi_level(calc, eps_skn=None, nelectrons=None):
    from gpaw.occupations import occupation_numbers
    from ase.units import Ha
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()
    if eps_skn is None:
        eps_skn = eigenvalues(calc)
    eps_skn.sort(axis=-1)
    occ = calc.occupations.todict()
    weight_k = calc.get_k_point_weights()
    return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha


def eigenvalues(calc):
    import numpy as np
    rs = range(calc.get_number_of_spins())
    rk = range(len(calc.get_ibz_k_points()))
    e = calc.get_eigenvalues
    return np.asarray([[e(spin=s, kpt=k) for k in rk] for s in rs])


def get_spin_direction(fname="anisotropy_xy.npz"):
    import os
    import numpy as np
    theta = 0
    phi = 0
    if os.path.isfile(fname):
        data = np.load(fname)
        DE = max(data["dE_zx"], data["dE_zy"])
        if DE > 0:
            theta = np.pi / 2
            if data["dE_zy"] > data["dE_zx"]:
                phi = np.pi / 2
    return theta, phi


group = 'Postprocessing'
dependencies = ['asr.anisotropy']


if __name__ == '__main__':
    main()
