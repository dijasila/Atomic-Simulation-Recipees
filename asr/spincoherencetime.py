from ase.io import read
from asr.core import command, ASRResult, prepare_result, option
import numpy as np


# def webpanel(result, row, key_description):
#
#     return None


@prepare_result
class Result(ASRResult):
    """Container for spin coherence time results."""

    coherence_function: np.ndarray

    key_descriptions = dict(
        coherence_function='Coherence function as a function of time [ms].')

    # formats = {'ase_webpanel': webpanel}


@command(module='asr.spincoherencetime',
         requires=['gs.gpw', 'structure.json'],
         dependencies=['asr.gs'],
         resources='1:1h',
         returns=Result)
@option('--pristinefile', help='Path to the pristine supercell file'
        '(needs to be of the same shape as structure.json).', type=str)
def main(pristinefile: str = 'pristine.json') -> Result:
    """Calculate spin coherence time."""
    defect = read('structure.json')
    pristine = read('pristinefile.json')

    # @SAJID: implement your embedding functionality in this function
    #         which just takes defect and pristine supercell as an input
    supercell = embed_supercell(defect, pristine)
    # @SAJID: what kind of input do you need to calculate the coherence
    #         function? Let's define that here and then implement the
    #         corresponding function for it.
    coherence_function = get_coherence_function(supercell)

    return Result.fromdata(
        coherence_function=coherence_function)


def get_coherence_function(supercell):
    """Calculate coherence function for a given input structure."""
    import numpy as np
    import pycce as pc
    from ase.io import read
    from pycce.bath.array import BathArray
    from pycce import common_concentrations
    from pathlib import Path
    from asr.core import read_json, write_json

    sr = read_json("fitted_Coherence_Times_Clean.json")
    pristinepath = []
    for p in range(len(sr["System"])):
        pristinepath.append(Path("./" + sr["System"][p]))
    seed = 80
    np.random.seed(seed)
    np.set_printoptions(suppress=True, precision=5)
    # Generate unitcell from ase
    for P in range(len(pristinepath)):
        if not (pristinepath[P] / 'Coherence-Final.json').is_file():
            str_1 = read(pristinepath[P] / 'structure.json')
            str_1 = pc.bath.BathCell.from_ase(str_1)
            str_1.zdir = [0, 0, 1]
            self = str_1
            size = 1
            if not self.isotopes:
                isotopes = {}
                for a in self.atoms:
                    try:
                        isotopes[a] = common_concentrations[a]
                    except KeyError:
                        pass
            else:
                isotopes = self.isotopes
            rgen = np.random.default_rng(seed)
            axb = np.cross(self.cell[:, 0], self.cell[:, 1])
            bxc = np.cross(self.cell[:, 1], self.cell[:, 2])
            cxa = np.cross(self.cell[:, 2], self.cell[:, 0])
            anumber = int(size * np.linalg.norm(bxc) / (bxc @ self.cell[:, 0]) + 1)
            bnumber = int(size * np.linalg.norm(cxa) / (cxa @ self.cell[:, 1]) + 1)
            cnumber = int(size * np.linalg.norm(axb) / (axb @ self.cell[:, 2]) + 1)
            # print(anumber, bnumber, cnumber)
            dt = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
            atoms = []
            for a in isotopes:
                nsites = len(self.atoms[a])
                # print(nsites)
                sites_xyz = np.asarray(self.atoms[a]) @ self.cell.T
                # print(sites_xyz)
                maxind = np.array([anumber,
                                   bnumber,
                                   cnumber,
                                   nsites], dtype=np.int32)
                natoms = np.prod(maxind, dtype=np.int32)
                atom_seedsites = np.arange(natoms, dtype=np.int32)
                mask = np.zeros(natoms, dtype=bool)
                for i in isotopes[a]:
                    conc = isotopes[a][i]
                    nisotopes = int(round(natoms * conc))
                    seedsites = rgen.choice(atom_seedsites[~mask],
                                            nisotopes, replace=False,
                                            shuffle=False)
                    mask += np.isin(atom_seedsites, seedsites)
                    bcn = anumber * bnumber * nsites
                    cn = cnumber * nsites
                    aindexes = seedsites // bcn - 1 / 2  # recenter at 0
                    bindexes = (seedsites % bcn) // cn - 1 / 2
                    cindexes = ((seedsites % bcn) % cn) // nsites
                    # indexes of the sites
                    nindexes = ((seedsites % bcn) % cn) % nsites
                    indexes = np.column_stack((aindexes,
                                               bindexes,
                                               cindexes))
                    uc_positions = np.einsum('jk,ik->ij', self.cell, indexes)
                    subatoms = np.zeros(indexes.shape[0], dtype=dt)
                    subatoms['N'] = i
                    subatoms['xyz'] = uc_positions + sites_xyz[nindexes]
                    atoms.append(subatoms)
            atoms = np.concatenate(atoms)
            # bath = bath[np.linalg.norm(bath['xyz'], axis=1) <= size]
    # defective_atoms = defect(self.cell, atoms, add=add, remove=remove)
            bath_222 = BathArray(array=atoms)
            atoms_3 = bath_222
    # atoms_3 = BN_3.gen_supercell(1, seed=seed)
    # Parameters of CCE calculations engine
    # Order of CCE aproximation
            order = 2
    # Bath cutoff radius
            r_bath = 40  # in A
    # Cluster cutoff radius
            r_dipole = 15  # in A
    # position of central spin
            position_3 = [2.5, 2.5, 350.0]
    # Qubit levels (in Sz basis)
            alpha = [0, 0, 1]
            beta = [0, 1, 0]
    # ZFS Parametters of NV center in diamond
            D = 2.00 * 1e6  # in kHz
            E = 0.0          # in kHz
            # @SAJID: this part is never used?
            # spin_types = [('14N', 1, 1.9338, 20.44),
            #               ('13C', 1 / 2, 6.72828),
            #               ('29Si', 1 / 2, -5.3188),
            #               ('10B', 3, 2.875),
            #               ('11B', 3 / 2, 8.584), ]
            calc_1 = pc.Simulator(spin=1, position=position_3,
                                  alpha=alpha, beta=beta, D=D, E=E,
                                  bath=atoms_3, r_bath=r_bath,
                                  r_dipole=r_dipole, order=order)

    # Time points
            time_space = np.linspace(0, 50, 5000)  # in ms
    # Number of pulses in CPMG seq (0 = FID, 1 = HE)
            N = 1
    # Mag. Field (Bx By Bz)
            B = np.array([0, 0, 50000])  # in G
            try:
                l_conv_1 = calc_1.compute(
                    time_space,
                    pulses=N,
                    magnetic_field=B,
                    method='cce',
                    quantity='coherence',
                    as_delay=False)
                dict = {
                    "Coherence": l_conv_1.real,
                    "Time(ms)": time_space,
                }
                print(
                    'INFO: writing the spin coherence of {} in folder {}.'.format(
                        pristinepath[P], pristinepath[P]))
                write_json(pristinepath[P] / 'Coherence-Final.json', dict)
            except Exception:
                pass
                print('WARNING: Coherence function could not be '
                      'computed for {} system.'.format(pristinepath[P]))

    # @SAJID: what should this function return? I guess a 2D array with times and
    #         coherence or something?
    return np.zeros((3, 3))


def embed_supercell(defect_sc, pris_sc):
    # @SAJID: implement your things here
    D = [defect_sc.get_cell()[0], defect_sc.get_cell()[1], [0.0, 0.0, 700]]
    # @SAJID: this might have changed, look into asr.get_wfs or asr.defect_symmetry
    #         to see the new version, and import it here accordingly
    # dd = -return_defect_coordinates(structure, unrelaxed, primitive, pristine, folder)
    dd = None  # update this in the new version
    defect_sc.translate(dd)
    pris_sc.translate(dd)
    defect_sc.set_cell(D)
    pris_sc.set_cell(D)
    defect_sc.translate([0.0, 0.0, 350])
    pris_sc.translate([0.0, 0.0, 350])
    d = len(pris_sc) - len(defect_sc)
    pris_large_sc = pris_sc.repeat((30, 30, 1))
    pos1 = pris_sc.get_positions()
    del pris_large_sc[0:len(pos1)]
    defective_bigger_sc = pris_large_sc + defect_sc
    assert len(defective_bigger_sc) == 900 * len(pris_sc) - d
    # defective_bigger_sc.translate([5, 5, 0.0])
    # defective_bigger_sc.wrap()
    # @SAJID: this should be removed
    # os.mkdir(f'/home/niflheim/sajal/MagmomConvergence/dummy/{name}/')
    # write(
    #     f'/home/niflheim/sajal/MagmomConvergence/dummy/{name}/structure.json'.format(name),
    #     defective_bigger_sc)

    return defective_bigger_sc


if __name__ == '__main__':
    main.cli()
