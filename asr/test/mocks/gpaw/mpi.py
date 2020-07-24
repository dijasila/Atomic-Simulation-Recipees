from types import SimpleNamespace


def broadcast(n, root):
    pass


def new_communicator(ranks):
    pass


def barrier():
    pass


def world_sum(a, **ignored):
    if isinstance(a, (int, float, complex)):
        return a
    else:
        pass


world = SimpleNamespace(size=1,
                        rank=0,
                        broadcast=broadcast,
                        barrier=barrier,
                        new_communicator=new_communicator,
                        sum=world_sum)

serial_comm = None

SerialCommunicator = SimpleNamespace
