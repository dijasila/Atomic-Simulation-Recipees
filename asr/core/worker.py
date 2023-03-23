from htwutil.worker import main


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
    from asr.core.repository import ASRRepository
    repo = ASRRepository.find()
    try:
        from gpaw.mpi import world
    except ModuleNotFoundError:
        comm = None
    else:
        comm = HTWCommunicatorWrapper(world)
    main(repo, comm=comm)
