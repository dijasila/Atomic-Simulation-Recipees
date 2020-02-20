from types import SimpleNamespace


def broadcast(n, root):
    pass


def new_communicator(ranks):
    pass


world = SimpleNamespace(size=1,
                        rank=0,
                        broadcast=broadcast,
                        new_communicator=new_communicator)

serial_comm = None
