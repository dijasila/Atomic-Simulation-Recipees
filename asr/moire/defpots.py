import typing
from gpaw import GPAW
from ase import Atoms
from ase.io import read
from ase.io.jsonio import read_json
import numpy as np
from asr.utils.bands import get_all_eigenvalues, calculate_evac, get_cb_vb_surface, plot_cb_vb_surface, calculate_evac
from asr.core import command, option, ASRResult, prepare_result
from ase.dft.bandgap import bandgap


def get_vacuum_level(filename: str = 'results-asr.gs.json', directory: str = '.'):
    from asr.gs import Result as r
    stuff = r.fromdict(read_json(f'{directory}/{filename}'))
    return stuff.evac


def get_relevant_strains(pbc):
    import numpy as np

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    if np.sum(pbc) == 3:
        ij = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    elif np.sum(pbc) == 2:
        ij = ((0, 0), (1, 1), (0, 1))
    elif np.sum(pbc) == 1:
        ij = ((2, 2), )

    return ij, [ij_to_voigt[i][j] for (i, j) in ij]


def make_strained_atoms(
        atoms: Atoms,
        strain_percent: float = 1,
        i: int = 0,
        j: int = 0,
):
    import numpy as np

    atoms = atoms.copy()
    cell_cv = atoms.get_cell()

    strain_vv = np.eye(3)
    strain_vv[i, j] += strain_percent / 100.0
    strain_vv = (strain_vv + strain_vv.T) / 2
    strained_cell_cv = np.dot(cell_cv, strain_vv)
    atoms.set_cell(strained_cell_cv, scale_atoms=True)

    return atoms


def make_strained_tree(atoms, strain_percent, directory='.'):
    from pathlib import Path
    Path(f'{directory}/strained').mkdir(parents=True)
    strain_ij, _ = get_relevant_strains(atoms.pbc)
    for i, j in strain_ij:
        for strain_perc in [-strain_percent, strain_percent]:
            strained_atoms = make_strained_atoms(atoms, strain_perc, i, j)
            dirname = f'{directory}/strained/{i}_{j}_{strain_perc}'
            Path(dirname).mkdir()
            strained_atoms.write(f'{dirname}/unrelaxed.json')


#TODO use finer parameters after testing!
# ALTERNATIVE: import gs@calculate, give custom parameters
def custom_gs(parms, atoms='structure.json', direc='.', outfile='gs.gpw'):
    atoms = read(f'{direc}/{atoms}')
    params: dict = {
        'mode': {'name': 'pw', 'ecut': 400},
        'xc': 'PBE',
        'basis': 'dzp',
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '200%',
        'charge': 0,
        'txt': '-'
    }
    params.update(parms)
    atoms.calc = GPAW(**params)
    atoms.get_potential_energy()
    atoms.calc.write(f'{direc}/{outfile}')
    return get_cb_vb_surface(atoms.calc)
    

@prepare_result
class Result(ASRResult):

    deformation_potentials_nosoc: typing.List[float]
    deformation_potentials_soc: typing.List[float]
    edges_nosoc: typing.List[float]
    edges_soc: typing.List[float]
    edges_kpin_nosoc: typing.List[float]
    edges_kpin_soc: typing.List[float]
    kpts: typing.List[float]

    key_descriptions = {'deformation_potentials_nosoc': 'Deformation potentials under different types \
                         of deformations (xx, yy, yz, xz, xy) at each k-point, without SOC',
                        'deformation_potentials_soc': 'Deformation potentials under different types \
                         of deformations (xx, yy, yz, xz, xy) at each k-point, with SOC',
                        'edges_nosoc': 'Valence and conduction band edges of the unstrained system at each k-point',
                        'edges_soc': 'Valence and conduction band edges of the unstrained system at each k-point',
                        'edges_kpin_nosoc': 'edges_kpin',
                        'edges_kpin_soc': 'edges_kpin',
                        'kpts': 'k-points at which deformation potentials were calculated'}


@command('asr.moire.defpots',
         returns=Result)
@option('-s', '--strain_percent', help='Strain percentage', type=float)
#@option('--no-restart', is_flag=True, help="Don't recalculate band edges at the special points", type=bool)
@option('--special-kpts-only', is_flag=True, help="Calculate deformation potentials only at the special points.", type=bool)
#@option('--soc', is_flag=True, help='Calculate spin-orbit coupling eigenvalues and corresponding deformation potentials', type=bool)
def main(strain_percent = 1.0,
         special_kpts_only = False) -> Result:

    def get_edges(calc, soc):
        from gpaw.spinorbit import soc_eigenstates
        if soc:
            soc = soc_eigenstates(calc)
            evs = soc.eigenvalues()
            ef = calc.get_fermi_level()
            _, cb, vb = get_cb_vb_surface(eigenvalues=evs, ef=ef, kpts=kpts, all_bz=False)
        else:
            _, cb, vb = get_cb_vb_surface(calc=calc, all_bz=False)
        return vb, cb

    # If requested, We obtain the deformation potentials at each special point (including Lambda)
    if special_kpts_only:
        calc = GPAW('gs.gpw')
        atoms = calc.get_atoms()
        specpts = atoms.cell.bandpath(pbc=atoms.pbc).special_points
        kpts = [specpts[i] for i in specpts.keys()]
        kpts.append([1/6, 1/6, 0.0])
        calc_nostrain = calc.fixed_density(kpts=kpts, symmetry='off', txt='-')
        calc_nostrain.get_potential_energy()
        calc_nostrain.write('gs_spec.gpw')

    # Otherwise we use all k-points in the irreducible Brillouin zone
    else:
        calc_nostrain = GPAW('gs.gpw')
        atoms = calc_nostrain.get_atoms()
        kpts = calc_nostrain.get_ibz_k_points()

    ij, comps = get_relevant_strains(atoms.pbc)
    results = {'kpts': kpts}

    # Band edges at different k points are now collected in edges_kpin, with shape:
    # (N_kpts, N_strain_percents (including 0%), N_strain_components, (vbm, cbm)).
    # We will have two sets of deformation potentials, i.e. with and without SOC.
    for soc, flag in zip([False, True], ['nosoc', 'soc']):

        vb, cb = get_edges(calc_nostrain, soc)
        edges_kpin = np.zeros((len(kpts), 3, 6, 2), float)
        edges_k = np.zeros((len(kpts), 2))

        # Filling in edges_kpin with the edges of the unstrained material
        evac = calculate_evac(calc_nostrain)
        if not soc:
            print(f'unstrained: {evac}')
        for k in range(len(kpts)):
            edges_k[k] = np.asarray([vb[k], cb[k]]) - evac
            for comp in comps:
                edges_kpin[k, 1, comp] = edges_k[k]

        # Cycling through directories to extract the edges of the strained material
        for ind, i in zip(ij, comps):
            for p, perc in zip([0, 2], [-strain_percent, strain_percent]):
                direc = f'strained/{ind[0]}_{ind[1]}_{perc}'
                gpw = GPAW(f'{direc}/gs.gpw')

                # This is just to perform the fixed density calculation only once
                if not soc:
                    calc = gpw.fixed_density(kpts=kpts, symmetry='off', txt='gs_spec.txt')
                    calc.get_potential_energy()
                    calc.write(f'{direc}/gs_spec.gpw')
                else:
                    calc = GPAW(f'{direc}/gs_spec.gpw')

                vb, cb = get_edges(calc, soc)
                evac = calculate_evac(calc)
                if not soc:
                    print(f'{perc}%, {ind}: {evac}')
                for k in range(len(kpts)):
                    edges_kpin[k, p, i, 0] = vb[k] - evac
                    edges_kpin[k, p, i, 1] = cb[k] - evac

        strains = np.asarray([-strain_percent, 0.0, strain_percent]) * 0.01
        deformation_potentials = np.zeros((len(kpts), 6, 2))
        for k, kpt in enumerate(kpts):
            for indx, band_edge in enumerate(['vbm', 'cbm']):
                #D = np.polyfit(strains, edges_kpin[k, :, :, indx], 1)[0]
                D = (edges_kpin[k, 2, :, indx] - edges_kpin[k, 0, :, indx]) / (strains[2] - strains[0])
                deformation_potentials[k, :, indx] = D

        results.update({f'edges_{flag}': edges_k.tolist(),
                        f'edges_kpin_{flag}': edges_kpin.tolist(),
                        f'deformation_potentials_{flag}': deformation_potentials.tolist()})
    return results


if __name__ == '__main__':
    main.cli()
