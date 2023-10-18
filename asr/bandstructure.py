"""Electronic band structures."""
from typing import Union
import numpy as np

from asr.core import command, option, ASRResult, singleprec_dict
from asr.utils.symmetry import c2db_symmetry_eps
from asr.paneldata import BandStructureResult


@command('asr.bandstructure',
         requires=['gs.gpw'],
         creates=['bs.gpw'],
         dependencies=['asr.gs@calculate'])
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints', type=int)
@option('--emptybands', type=int)
def calculate(kptpath: Union[str, None] = None, npoints: int = 400,
              emptybands: int = 20) -> ASRResult:
    """Calculate electronic band structure."""
    from gpaw import GPAW
    from ase.io import read

    atoms = read('structure.json')
    path = atoms.cell.bandpath(path=kptpath, npoints=npoints,
                               pbc=atoms.pbc, eps=c2db_symmetry_eps)

    convbands = emptybands // 2
    parms = {
        'basis': 'dzp',
        'nbands': -emptybands,
        'txt': 'bs.txt',
        'fixdensity': True,
        'kpts': path,
        'convergence': {
            'bands': -convbands},
        'symmetry': 'off'}
    calc = GPAW('gs.gpw', **parms)
    calc.get_potential_energy()
    calc.write('bs.gpw')


@command('asr.bandstructure',
         requires=['gs.gpw', 'bs.gpw', 'results-asr.gs.json',
                   'results-asr.structureinfo.json',
                   'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.bandstructure@calculate', 'asr.gs',
                       'asr.structureinfo', 'asr.magnetic_anisotropy'],
         returns=BandStructureResult)
def main() -> BandStructureResult:
    from gpaw import GPAW
    from ase.spectrum.band_structure import get_band_structure
    from ase.dft.kpoints import BandPath
    from asr.core import read_json
    import copy
    from asr.utils.gpw2eigs import gpw2eigs
    from asr.magnetic_anisotropy import get_spin_axis, get_spin_index

    ref = GPAW('gs.gpw', txt=None).get_fermi_level()
    calc = GPAW('bs.gpw', txt=None)
    atoms = calc.atoms
    path = calc.parameters.kpts
    if not isinstance(path, BandPath):
        if 'kpts' in path:
            # In this case path comes from a bandpath object
            path = BandPath(kpts=path['kpts'], cell=path['cell'],
                            special_points=path['special_points'],
                            path=path['labelseq'])
        else:
            path = calc.atoms.cell.bandpath(pbc=atoms.pbc,
                                            path=path['path'],
                                            npoints=path['npoints'],
                                            eps=c2db_symmetry_eps)
    bs = get_band_structure(calc=calc, path=path, reference=ref)

    results = {}
    bsresults = bs.todict()

    # Save Fermi levels
    gsresults = read_json('results-asr.gs.json')
    efermi_nosoc = gsresults['gaps_nosoc']['efermi']
    bsresults['efermi'] = efermi_nosoc

    # We copy the bsresults dict because next we will add SOC
    results['bs_nosoc'] = copy.deepcopy(bsresults)  # BS with no SOC

    # Add spin orbit correction
    bsresults = bs.todict()

    theta, phi = get_spin_axis()

    # We use a larger symmetry tolerance because we want to correctly
    # color spins which doesn't always happen due to slightly broken
    # symmetries, hence tolerance=1e-2.
    e_km, _, s_kvm = gpw2eigs(
        'bs.gpw', soc=True, return_spin=True, theta=theta, phi=phi,
        symmetry_tolerance=1e-2)
    bsresults['energies'] = e_km.T
    efermi = gsresults['efermi']
    bsresults['efermi'] = efermi

    # Get spin projections for coloring of bandstructure
    path = bsresults['path']
    npoints = len(path.kpts)
    s_mvk = np.array(s_kvm.transpose(2, 1, 0))

    if s_mvk.ndim == 3:
        sz_mk = s_mvk[:, get_spin_index(), :]  # take x, y or z component
    else:
        sz_mk = s_mvk

    assert sz_mk.shape[1] == npoints, f'sz_mk has wrong dims, {npoints}'

    bsresults['sz_mk'] = sz_mk

    return BandStructureResult.fromdata(
        bs_soc=singleprec_dict(bsresults),
        bs_nosoc=singleprec_dict(results['bs_nosoc'])
    )


if __name__ == '__main__':
    main.cli()
