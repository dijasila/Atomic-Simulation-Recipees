from asr.core import command, option, argument, ASRResult, prepare_result
from asr.utils.symmetry import c2db_symmetry_eps
from typing import List, Union
import numpy as np
from os import path


@command(module='asr.spinspiral',
         requires=['structure.json'])
@argument('q_c', type=List[float])
@option('--n', type=int)
@option('--params', help='Calculator parameter dictionary', type=dict)
@option('--smooth', help='Rotate initial magmoms by q dot a', type=bool)
def calculate(q_c : List[float] = [1 / 3, 1 / 3, 0], n : int = 0,
              params: dict = dict(mode={'name': 'pw', 'ecut': 400},
                                  kpts={'density': 4.0, 'gamma': True}),
              smooth: bool = True) -> ASRResult:
    """Calculate the groundstate of a given spin spiral vector q_c"""

    from ase.io import read
    from gpaw import GPAW
    atoms = read('structure.json')
    restart = path.isfile(f'gsq{n}.gpw')

    # IF the calculation has been run (gs.txt exist) but did not converge
    # (gs.gpw not exist) THEN raise exception UNLESS it did not finish (last calc)
    # (gsn.txt exist but gsn+1.txt does not)
    # Note, UNLESS imply negation in context of if-statement (and not)
    if path.isfile(f'gsq{n}.txt') and not path.isfile(f'gsq{n}.gpw') and \
       not (path.isfile(f'gsq{n}.txt') and not path.isfile(f'gsq{n+1}.txt')):
        raise Exception("SFC finished but didn't converge")

    try:
        try:
            magmoms = params["magmoms"]
        except KeyError:
            magmoms = params["experimental"]["magmoms"]
    except KeyError:
        if atoms.has('initial_magmoms'):
            magmomx = atoms.get_initial_magnetic_moments()
        else:
            magmomx = np.ones(len(atoms), float)
        magmoms = np.zeros((len(atoms), 3))
        magmoms[:, 0] = magmomx

        if smooth:  # Smooth spiral
            def rotate_magmoms(magmoms, q):
                import numpy as np
                from ase.io import read
                from ase.dft.kpoints import kpoint_convert

                def rotation_matrix(axis, theta):
                    """
                    Return the rotation matrix associated with counterclockwise rotation
                    about the given axis by theta radians.
                    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
                    """
                    axis = np.asarray(axis)
                    axis = axis / np.sqrt(np.dot(axis, axis))
                    a = np.cos(theta / 2.0)
                    b, c, d = -axis * np.sin(theta / 2.0)
                    aa, bb, cc, dd = a * a, b * b, c * c, d * d
                    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
                    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
                atoms = read('structure.json')
                q_v = kpoint_convert(atoms.get_cell(), skpts_kc=[q])[0]
                pos_av = atoms.get_positions()
                theta = np.dot(pos_av, q_v)
                R = [rotation_matrix([0, 0, 1], theta[i]) for i in range(len(atoms))]
                magmoms = [R[i] @ magmoms[i] for i in range(len(atoms))]
                return np.asarray(magmoms)

            magmoms = rotate_magmoms(magmoms, q_c)

    if restart:
        params = dict(restart=f'gsq{n}.gpw')
    else:
        # Mandatory spin spiral parameters
        params["mode"]["qspiral"] = q_c
        params["xc"] = 'LDA'
        params["magmoms"] = magmoms
        params["soc"] = False
        params["symmetry"] = 'off'
        params["parallel"] = {'domain': 1, 'band': 1}
        params["txt"] = f'gsq{n}.txt'

    calc = GPAW(**params)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    totmom_v, magmom_av = calc.density.state.density.calculate_magnetic_moments()

    if not restart:
        atoms.calc.write(f'gsq{n}.gpw')
    return ASRResult.fromdata(en=energy, q=q_c, ml=magmom_av, mT=totmom_v)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig
    spiraltable = table(row, 'Property', ['bandwidth', 'minimum'], key_descriptions)

    panel = {'title': 'Spin spirals',
             'columns': [[fig('spin_spiral_bs.png')], [spiraltable]],
             'plot_descriptions': [{'function': plot_bandstructure,
                                    'filenames': ['spin_spiral_bs.png']}],
             'sort': 3}
    return [panel]


@prepare_result
class Result(ASRResult):
    path: np.ndarray
    energies: np.ndarray
    local_magmoms: np.ndarray
    total_magmoms: np.ndarray
    bandwidth: float
    minimum: np.ndarray
    key_descriptions = {"path": "List of Spin spiral vectors",
                        "energies": "Potential energy [eV]",
                        "local_magmoms": "List of estimated local moments [mu_B]",
                        "total_magmoms": "Estimated total moment [mu_B]",
                        "bandwidth": "Energy difference [meV]",
                        "minimum": "Q-vector at energy minimum"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.spinspiral',
         requires=['structure.json'],
         returns=Result)
@option('--q_path', help='Spin spiral high symmetry path eg. "GKMG"', type=str)
@option('--n', type=int)
@option('--params', help='Calculator parameter dictionary', type=dict)
@option('--smooth', help='Rotate initial magmoms by q dot a', type=bool)
@option('--eps', help='Bandpath symmetry threshold', type=float)
def main(q_path: Union[str, None] = None, n: int = 11,
         params: dict = dict(mode={'name': 'pw', 'ecut': 600},
                             kpts={'density': 6.0, 'gamma': True}),
         smooth: bool = True, eps: float = None) -> Result:
    from ase.io import read
    atoms = read('structure.json')
    cell = atoms.cell
    if eps is None:
        eps = c2db_symmetry_eps

    if q_path is None:
        # Input --q_path None
        # eps = 0.1 is current c2db threshold
        path = atoms.cell.bandpath(npoints=n, pbc=atoms.pbc, eps=eps)
        Q = np.round(path.kpts, 16)
    elif q_path == 'ibz':
        # Input: --q_path 'ibz'
        from gpaw.symmetry import atoms2symmetry
        from gpaw.kpt_descriptor import kpts2sizeandoffsets
        from ase.dft.kpoints import monkhorst_pack, kpoint_convert
        # Create (n,n,1) for 2D, (n,n,n) for 3D
        sizeInput = atoms.get_pbc() * (n - 1) + 1
        size, offset = kpts2sizeandoffsets(sizeInput, gamma=True, atoms=atoms)
        bzk_kc = monkhorst_pack(size) + offset
        symmetry = atoms2symmetry(atoms, tolerance=eps)
        ibzinfo = symmetry.reduce(bzk_kc)
        Q = ibzinfo[0]
        bz2ibz_k, ibz2bz_k = ibzinfo[4:6]
        bzk_kv = kpoint_convert(cell, skpts_kc=bzk_kc) / (2 * np.pi)
        path = [bzk_kv, bz2ibz_k, ibz2bz_k]
    elif q_path.isalpha():
        # Input: --q_path 'GKMG'
        path = atoms.cell.bandpath(q_path, npoints=n, pbc=atoms.pbc, eps=eps)
        Q = np.round(path.kpts, 16)
    else:
        # Input: --q_path 111 --n 5
        from ase.dft.kpoints import (monkhorst_pack, kpoint_convert)
        import sys
        sys.path.insert(0, "/home/niflheim/joaso/scripts/")
        from rotation import project_to_plane
        plane = [eval(q) for q in q_path]
        Q = monkhorst_pack([n, n, n])
        Q = project_to_plane(Q, plane)
        Qv = kpoint_convert(atoms.get_cell(), skpts_kc=Q) / (2 * np.pi)
        path = [Q, Qv]

    energies = []
    lmagmom_av = []
    Tmagmom_v = []
    for i, q_c in enumerate(Q):
        try:
            result = calculate(q_c=q_c, n=i, params=params, smooth=smooth)
            energies.append(result['en'])
            lmagmom_av.append(result['ml'])
            Tmagmom_v.append(result['mT'])
        except Exception as e:
            print('Exception caught: ', e)
            energies.append(0)
            lmagmom_av.append(np.zeros((len(atoms), 3)))
            Tmagmom_v.append(np.zeros(3))

    energies = np.asarray(energies)
    lmagmom_av = np.asarray(lmagmom_av)
    Tmagmom_v = np.asarray(Tmagmom_v)

    bandwidth = (np.max(energies) - np.min(energies)) * 1000
    qmin = Q[np.argmin(energies)]
    return Result.fromdata(path=path, energies=energies,
                           local_magmoms=lmagmom_av, total_magmoms=Tmagmom_v,
                           bandwidth=bandwidth, minimum=qmin)


def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    data = row.data.get('results-asr.spinspiral.json')
    path = data['path']
    energies = data['energies']

    energies = ((energies - energies[0]) * 1000)  # / nmagatoms
    q, x, X = path.get_linear_kpoint_axis()

    total_magmoms = data['total_magmoms']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Setup main energy plot
    ax1.plot(q, energies, c='C1', marker='.', label='Energy')
    ax1.set_ylim([np.min(energies * 1.1), np.max(energies * 1.15)])
    ax1.set_ylabel('Spin spiral energy [meV]')

    ax1.set_xlabel('q vector [Å$^{-1}$]')
    ax1.set_xticks(x)
    ax1.set_xticklabels([i.replace('G', r"$\Gamma$") for i in X])
    for xc in x:
        if xc != min(q) and xc != max(q):
            ax1.axvline(xc, c='gray', linestyle='--')
    ax1.margins(x=0)

    # Add spin wavelength axis
    def tick_function(X):
        lmda = 2 * np.pi / X
        return [f"{z:.1f}" for z in lmda]

    # Non-cumulative length of q-vectors to find wavelength
    Q = np.linalg.norm(2 * np.pi * path.cartesian_kpts(), axis=-1)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    idx = round(len(Q) / 5)

    ax2.set_xticks(q[::idx])
    ax2.set_xticklabels(tick_function(Q[::idx]))
    ax2.set_xlabel(r"Wave length $\lambda$ [Å]")

    # Add the magnetic moment plot
    ax3 = ax1.twinx()
    mT = abs(total_magmoms[:, 0])
    # mT = np.linalg.norm(total_magmoms, axis=-1)#mT[:, 1]#
    mT2 = abs(total_magmoms[:, 1])
    mT3 = abs(total_magmoms[:, 2])
    ax3.plot(q, mT, c='r', marker='.', label='$m_x$')
    ax3.plot(q, mT2, c='g', marker='.', label='$m_y$')
    ax3.plot(q, mT3, c='b', marker='.', label='$m_z$')

    ax3.set_ylabel(r"Total norm magnetic moment ($\mu_B$)")
    mommin = np.min(mT * 0.9)
    mommax = np.max(mT * 1.15)
    ax3.set_ylim([mommin, mommax])

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # fig.suptitle('')
    plt.tight_layout()
    plt.savefig(fname)

    # energies = energies - energies[0]
    # energies = (energies)*1000
    # bs = BandStructure(path=path, energies=energies[None, :, None])
    # bs.plot(ax=plt.gca(), ls='-', marker='.', colors=['C1'],
    #         emin=np.min(energies * 1.1), emax=np.max([np.max(energies * 1.15)]),
    #         ylabel='Spin spiral energy [meV]')
    # plt.tight_layout()
    # plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
