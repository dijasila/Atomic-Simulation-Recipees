from .. import GPAW


class BSE:

    def __init__(self, calc=None, *args, **kwargs):
        self.calc = GPAW(calc)

    def calculate(self):
        pass

    def get_polarizability(self, w_w=None, eta=0.1,
                           q_c=[0.0, 0.0, 0.0], direction=0,
                           filename='pol_bse.csv', readfile=None, pbc=None,
                           write_eig='eig.dat'):

        # world.rank == 0 and
        if filename is not None:
            fd = open(filename, 'w')
            for iw, w in enumerate(w_w):
                print('%.9f, %.9f, %.9f' %
                      (w, 1.0, 0.0), file=fd)
            fd.close()

        if write_eig is not None:
            f = open(write_eig, 'w')
            print('# %s eigenvalues in eV' % 'BSE', file=f)
            for iw in range(100):
                print('%8d %12.6f %12.16f' % (iw, iw * 0.1, 1.0),
                      file=f)
            f.close()

        return w_w, w_w
