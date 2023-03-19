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


if __name__ == '__main__':
    repo = ASRRepository.find()
    main(repo, comm=HTWCommunicatorWrapper(world))
