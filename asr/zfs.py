from asr.core import command
from asr.result.resultdata import ZfsResult


@command(module='asr.zfs',
         requires=['gs.gpw', 'structure.json'],
         dependencies=['asr.gs'],
         resources='1:1h',
         returns=ZfsResult)
def main() -> ZfsResult:
    """Calculate zero-field-splitting."""
    from gpaw import restart

    # read in atoms and calculator, check whether spin and magnetic
    # moment are suitable, i.e. spin-polarized triplet systems
    atoms, calc = restart('gs.gpw', txt=None)
    magmom = atoms.get_magnetic_moment()
    _ = check_magmoms(magmom)

    # run fixed density calculation
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})

    # evaluate zero field splitting components for both spin channels
    D_vv = get_zfs_components(calc)

    return ZfsResult.fromdata(
        D_vv=D_vv)


def get_zfs_components(calc):
    """Get ZFS components from GPAW function, and convert tensor scale."""
    from ase.units import _e, _hplanck
    from gpaw.zero_field_splitting import convert_tensor, zfs

    D_vv = zfs(calc)
    scale = _e / _hplanck * 1e-6
    D, E, axis, D_vv = convert_tensor(D_vv * scale)

    return D_vv


def check_magmoms(magmom, target_magmom=2, threshold=1e-1):
    """Check whether input atoms are a triplet system."""
    if not abs(magmom - target_magmom) < threshold:
        raise ValueError('ZFS recipe only working for systems with '
                         f'a total magnetic moment of {target_magmom}!')


if __name__ == '__main__':
    main.cli()
