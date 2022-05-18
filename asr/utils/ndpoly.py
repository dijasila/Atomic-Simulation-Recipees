from itertools import combinations_with_replacement
import numpy as np


def deriv(f, d):
    return [f[:i] + f[i + 1:] for i, x in enumerate(f) if x == d]


def tuples2str(tuples):
    if not tuples:
        return '0'
    assert len(set(tuples)) == 1
    return '*'.join([str(len(tuples))] + ['xyz'[d] for d in tuples[0]])


class PolyFit:
    def __init__(self, x, y, order=2, verbose=False):
        ndims = x.shape[1]
        self.ndims = ndims

        t0 = []
        for n in range(order + 1):
            t0.extend(combinations_with_replacement(range(ndims), n))
        args = ', '.join('xyz'[:ndims])
        s0 = ', '.join(tuples2str([t]) for t in t0)
        self.f0 = eval(compile(f'lambda {args}: [{s0}]', '', 'eval'))

        t1 = [[deriv(t, d) for t in t0] for d in range(ndims)]
        s1 = '], ['.join(', '.join(tuples2str(tt)
                                   for tt in t1[d])
                         for d in range(ndims))
        self.f1 = eval(compile(f'lambda {args}: [[{s1}]]', '', 'eval'))

        t2 = [[[sum((deriv(t, d2) for t in tt), start=[]) for tt in t1[d1]]
               for d1 in range(ndims)]
              for d2 in range(ndims)]
        s2 = ']], [['.join('], ['.join(', '.join(tuples2str(tt)
                                                 for tt in t2[d1][d2])
                                       for d1 in range(ndims))
                           for d2 in range(ndims))
        self.f2 = eval(compile(f'lambda {args}: [[[{s2}]]]', '', 'eval'))

        M = self.f0(*x.T)
        M[0] = np.ones(len(x))
        M = np.array(M)
        self.coefs = np.linalg.solve(M @ M.T, M @ y)

        if verbose:
            print(f'[{s0}]')
            print(f'[[{s1}]]')
            print(f'[[[{s2}]]]')
            print(self.coefs)

    def value(self, k_v):
        return self.f0(*k_v) @ self.coefs

    def gradient(self, k_v):
        return self.f1(*k_v) @ self.coefs

    def hessian(self, k_v):
        return self.f2(*k_v) @ self.coefs

    def find_minimum(self, k_v=None):
        from scipy.optimize import minimize

        def f(k_v):
            return self.value(k_v), self.gradient(k_v)

        if k_v is None:
            k_v = np.zeros(self.ndims)

        result = minimize(f, k_v, jac=True, method='Newton-CG')
        return result.x
