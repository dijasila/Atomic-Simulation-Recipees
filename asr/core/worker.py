from htwutil.worker import main
from asr.core.repository import ASRRepository
from gpaw.mpi import world


class HTWCommunicatorWrapper:
    def __init__(self, comm):
        self._comm = comm
        self.rank = comm.rank
        self.size = comm.size

    def broadcast_object(self, obj):
        from ase.parallel import broadcast
        return broadcast(obj, root=0, comm=self._comm)

    def split(self, newsize):
        import numpy as np
        assert self.size % newsize == 0

        mygroup = self.rank // newsize

        startrank = mygroup * newsize
        endrank = startrank + newsize
        ranks = np.arange(startrank, endrank)
        _comm = self._comm.new_communicator(ranks)
        assert _comm is not None
        return HTWCommunicatorWrapper(_comm)

    def mpicomm(self):
        return self._comm


if __name__ == '__main__':
    repo = ASRRepository.find()
    main(repo, comm=HTWCommunicatorWrapper(world))
