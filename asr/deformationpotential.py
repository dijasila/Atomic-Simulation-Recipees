from asr.core import command, option
import numpy as np
from gpaw import GPAW


@command()
@option('--strain-percent')
def main(gpw='densk.gpw',
         strains=[-1.0, 1.0],
         lattice_relax=False,
         ionic_relax=True,
         data={}):
    """
    Calculate the deformation potential both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary.
    """
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)
    strains = sorted(strains)
    atoms = read('structure.json')
    ij = get_relevant_strains(atoms.pbc)

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    strains = [0]
    edges = np.zeros((6, 3, 2), float) + np.nan

    for i, j in ij:
        vbm_i = np.zeros((1, 6), float) + np.nan
        vbmnosoc_i = np.zeros((1, 6), float) + np.nan
        cbm_i = np.zeros((1, 6), float) + np.nan
        cbmnosoc_i = np.zeros((1, 6), float) + np.nan
        for strain in [-1.0, 0.0, 1.0]:
            folder = get_strained_folder_name(strain, i, j)
            gsresults = read_json(folder / 'results-asr.gs.json')
            evac = gsresults['evac']
            edges[ij_to_voigt[i][j]][0] = gsresults['vbm'] - evac
            edges[ij_to_voigt[i][j]][0] = gsresults['gaps_nosoc']['vbm'] - evac
            cbm_i[ij_to_voigt[i][j]] = gsresults['cbm'] - evac
            cbmnosoc_i[ij_to_voigt[i][j]] = (gsresults['gaps_nosoc']['cbm'] -
                                             evac)

        edges[ij_to_voigt[i][j]][0] = vbm
        
    new_data = {'strains': np.array(strains)}
        for soc in (True, False):
            edges = np.array(edges)
            deformation_potentials = np.zeros(np.shape(edges)[1:])
            for idx, band_edge in enumerate(['vbm', 'cbm']):
                D = np.polyfit(strains, edges[:, :, idx], 1)[0] * 100
                D[2] /= 2
                deformation_potentials[:, idx] = D
        new_data[['edges_nosoc', 'edges'][soc]] = edges
        new_data[['deformation_potentials_nosoc',
                  'deformation_potentials'][soc]] = deformation_potentials
    data.update(**new_data)
    return data


def calculate_strained_edges(gpw='densk.gpw',
                             strain_percent=1.0,
                             gap_filename=None,
                             lattice_relax=False,
                             ionic_relax=True,
                             soc=True):
    """
    Calculate the band edge energies of a system after straining in the xx, the
    yy and the xxyy directions.

    params:
        gpw (str): The filename of the gpw restart file of a ground state
            calculation of the unstrained system.
        strain_percent (float): How much to strain the system by
        gap_filename (str): The filename of band edge information for the
            unstrained system. If this is None, the function looks in a
            standard location governed by ``soc''.
        soc (bool): Whether to use spin orbit coupling or not.
    returns:
        edges: a 3x2 numpy array containing the energy of the (VBM, CBM) with
            respect to vacuum for strain in the x direction, strain in y and
            biaxial strain.
    """

    from asr.core import read_json

    gsresults = read_json('results-asr.gs.json')
    if gap_filename is not None:
        data = np.load(gap_filename)
    elif soc:
        data = np.load('gap_soc.npz')
    else:
        data = np.load('gap.npz')
    if data['gap'] == 0.0:
        return None
    vbm = data['vbm']
    cbm = data['cbm']
    edges = np.zeros((3, 2))
    skn1 = np.array(data['skn1'])
    skn2 = np.array(data['skn2'])
    calc = GPAW(gpw, txt=None)
    if strain_percent == 0.0:
        # The pristine system and the different strained systems have slightly
        # different vacuum levels.
        old_vacuum = calc.get_electrostatic_potential().mean(0).mean(0)[0]
        edges[:, 0] = vbm - old_vacuum
        edges[:, 1] = cbm - old_vacuum
        return edges
    if soc:
        _, _, s_kvm = gpw2eigs(gpw, soc=True, return_spin=True,
                               optimal_spin_direction=True)
        s1, k1, n1 = skn1
        s2, k2, n2 = skn2
        sz1 = s_kvm[k1, 2, n1]
        sz2 = s_kvm[k2, 2, n2]
        skn1, skn2 = skn1.astype(float), skn2.astype(float)
        skn1[0], skn2[0] = sz1, sz2

    # The strained calculation can have a lower symmetry than the pristine one.
    # Therefore, the mappings BZ<->IBZ can be different, even with the same k
    # point sampling. We therefore get the BZ_k index, and use that to identify
    # the band edges.
    skn1[1] = calc.wfs.kd.ibz2bz_k[int(skn1[1])]
    skn2[1] = calc.wfs.kd.ibz2bz_k[int(skn2[1])]

    for idx, direction in enumerate(['x', 'y', 'xy']):
        new_gpw = strained_gpw_name(gpw, strain_percent, direction)
        if not os.path.isfile(new_gpw):
            calculate_strained_energies(gpw,
                                        strain_percent,
                                        direction,
                                        lattice_relax=lattice_relax,
                                        ionic_relax=ionic_relax)
        strained_energies = get_matching_energies(new_gpw, skn1, skn2, soc=soc)
        edges[idx] = np.array(strained_energies)
    return edges


def get_matching_energies(gpw, skn1, skn2, soc=False):
    """
    Find the energy of the band corresponding to a given
    k-point, band index and sz expectation value

    params:
        calc: A GPAW calculator object with the desired wavefunctions.
        skn1 (float, int, int): the expectation value of <sz>, the band
            index, n, and the kpt index of the vbm.
        skn2 (float, int, int): the expectation value of <sz>, the band
            index, n, and the kpt index of the cbm.
        soc: Whether or not to use spin orbit coupling. If not False, it should
            contain an array of spin orbit eigenvalues and <sz> expectation
            values.
    returns:
        energies (float, float): The energies of the new calculator at the
            spin, band and kpt indices corresponding to skn1 and skn2
            respectively.
    """

    from gpaw import GPAW
    from asr.utils.gpw2eigs import gpw2eigs
    energies = []
    calc = GPAW(gpw, txt=None)
    e_vac = calc.get_electrostatic_potential().mean(0).mean(0)[0]
    if soc:
        e_km, _, s_kvm = gpw2eigs(gpw, soc=True, return_spin=True,
                                  optimal_spin_direction=True)
        for idx, skn in enumerate([skn1, skn2]):
            sz_value, kpt, band = skn
            band, kpt = int(band), int(kpt)
            kpt = calc.wfs.kd.bz2ibz_k[kpt]
            # check the target band, as well as one higher (for cbm), or one
            # lower (for vbm).
            target_bands = (band, band - 1 + 2 * idx)
            new_energies = e_km[kpt, target_bands]
            new_spins = s_kvm[kpt, 2, target_bands]
            spin_index = np.abs(new_spins - sz_value).argmin()
            energy = new_energies[spin_index]
            energies.append(energy)
    else:
        for idx, skn in enumerate([skn1, skn2]):
            s, k, n = skn
            k = calc.wfs.kd.bz2ibz_k[k]
            e_n = calc.get_eigenvalues(kpt=k, spin=s)
            energy = e_n[n]
            energies.append(energy)
    return np.array(energies) - e_vac


