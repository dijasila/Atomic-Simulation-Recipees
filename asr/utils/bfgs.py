import time

import numpy as np
import ase.optimize
from ase.constraints import ExpCellFilter


class BFGS:
    def __init__(self, atoms,
                 fmax, smax, smask=None, emin=-np.inf,
                 logfile=None, trajectory=None):
        self.atoms = atoms
        self._logfile = logfile
        self.trajectory = trajectory

        self.smax = smax
        self.smask = smask
        self.fmax = fmax
        self.emin = emin
        uf = ExpCellFilter(self.atoms, mask=smask)

        self.opt = ase.optimize.BFGS(uf,
                                     logfile=self._logfile,
                                     trajectory=self.trajectory)
        self.opt.log = self.log
        self.opt.converged = self.converged
        self.force_consistent = self.opt.force_consistent
        self.step0 = self.opt.step
        self.opt.step = self.step

    def run(self):
        return self.opt.run(self.fmax)

    def step(self, f=None):
        m = self.atoms.get_magnetic_moments()
        self.step0(f)
        self.atoms.set_initial_magnetic_moments(m)

    def converged(self, forces=None):
        if self.atoms.get_potential_energy() < self.emin:
            return True
        forces = self.atoms.get_forces()  # get the raw forces
        if (forces**2).sum(axis=1).max() > self.opt.fmax**2:
            return False
        stress = self.atoms.get_stress() * self.smask
        return abs(stress).max() < self.smax

    @property
    def nsteps(self):
        return self.opt.nsteps

    @property
    def logfile(self):
        return self.opt.logfile

    def log(self, forces=None):
        forces = self.atoms.get_forces()  # get the raw forces
        fmax = (forces**2).sum(axis=1).max()**0.5
        stress = self.atoms.get_stress() * self.smask
        smax = abs(stress).max()
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)
        m = self.atoms.get_magnetic_moment()
        ms = self.atoms.get_magnetic_moments()
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                self.logfile.write(
                    '%s  %4s %8s %15s  %12s %12s %8s %8s\n' %
                    (' ' * len(name), 'Step',
                     'Time', 'Energy', 'fmax', 'smax', 'totmm', 'maxmm'))
                if self.force_consistent:
                    self.logfile.write(
                        '*Force-consistent energies used in optimization.\n')
            self.logfile.write(
                '%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f %12.4f '
                '%8.4f %8.4f\n' %
                (name, self.nsteps, T[3], T[4], T[5], e,
                 {1: '*', 0: ''}[self.force_consistent],
                 fmax, smax, m, abs(ms).max()))
            self.logfile.flush()
