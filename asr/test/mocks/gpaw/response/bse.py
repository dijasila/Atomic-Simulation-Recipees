from .. import GPAW


class BSE:

    def __init__(self, calc=None, *args, **kwargs):
        self.calc = GPAW(calc)

    def calculate(self):
        pass
