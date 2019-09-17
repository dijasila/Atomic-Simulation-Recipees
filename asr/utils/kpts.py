from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so


def get_kpts_size(atoms, density):
    """Try to get a reasonable Monkhorst-Pack grid which hits
    high symmetry points
    """
    size, offset = k2so(atoms=atoms, density=density)
    # size[2] = 1  # what do we do in general? XXX
    for i in range(len(size)):
        if size[i] % 6 != 0 and size[i] != 1:  # works for hexagonal cells XXX
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts
