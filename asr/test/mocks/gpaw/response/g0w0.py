from .. import GPAW


class G0W0:

    def __init__(self, calc=None, *args, **kwargs):
        self.calc = GPAW(calc)

    def calculate(self):
        pass
