def get_kpts_size(atoms, kptdensity):
    """Try to get a reasonable Monkhorst-Pack grid which hits high symmetry points."""
    from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
    size, offset = k2so(atoms=atoms, density=kptdensity)

    # XXX Should fix kpts2sizeandoffsets
    for i in range(3):
        if not atoms.pbc[i]:
            size[i] = 1
            offset[i] = 0

    for i in range(len(size)):
        if size[i] % 6 != 0 and size[i] != 1:  # works for hexagonal cells XXX
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts
