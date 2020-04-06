from types import SimpleNamespace


def broadcast(n, root):
    pass


def new_communicator(ranks):
    pass


def barrier():
    pass


world = SimpleNamespace(size=1,
                        rank=0,
                        broadcast=broadcast,
                        barrier=barrier,
                        new_communicator=new_communicator)

serial_comm = None

SerialCommunicator = SimpleNamespace
