from types import SimpleNamespace


def broadcast(n, root):
    pass


world = SimpleNamespace(size=1,
                        rank=0,
                        broadcast=broadcast)

serial_comm = None
