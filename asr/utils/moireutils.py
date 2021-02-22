from ase import Atoms
from typing import Union


def get_layers(atoms):
    copy = atoms.copy()
    top = copy[copy.get_tags() == 1]
    bottom = copy[copy.get_tags() == 0]
    if len(top) == 0:
        raise ValueError('Tags have not been set for the current structure. Aborting')
    return top, bottom


class Bilayer(Atoms):
    def __init__(self, atoms):
        super().__init__(atoms)
        self.top_layer = get_layers(atoms)[0]
        self.bottom_layer = get_layers(atoms)[1]
        self._atoms = atoms

    def copy(self):
        '''Returns a copy of itself.

        This is necessary because for some reason the copy() method
        inherited from the Atoms class fails (unexpected keyword argument 'cell')
        '''
        return self._atoms.copy()

    def get_total_thickness(self):
        zvals = self.positions[:, 2]
        return zvals.max() - zvals.min()

    def get_interlayer_distance(self):
        ceiling = self.top_layer.positions[:, 2].min()
        floor = self.bottom_layer.positions[:, 2].max()
        distance = ceiling - floor
        if distance < 0:
            raise ValueError('Your top and bottom layer are inverted!')
        return distance

    def set_interlayer_distance(self, distance: float):
        current_distance = self.get_interlayer_distance()
        shift = distance - current_distance
        bottom = self.bottom_layer.copy()
        top = self.top_layer.copy()
        top.translate([0, 0, shift])
        new = top + bottom
        self.__init__(new)

    def sort_along_z(self, order: Union[list, str] = 'descending'):
        '''Returns a copy of the bilayer,
           sorted in descending or ascending order
           with respect to the atom coordinates along z

        It's also possible to specify a custom order through a list
        '''
        import numpy as np

        sorter = np.argsort(self.positions[:, 2])
        old = self._atoms.copy()
        new = old[sorter]
        new = old[sorter]
        if order == 'descending':
            new = new[::-1]
        elif isinstance(order, list):
            new = old[order]

        self.__init__(new)

    def set_vacuum(self, value):
        oldcell = self.cell
        thick = self.get_total_thickness()
        new_z = thick + value
        newcell = [oldcell[0],
                   oldcell[1],
                   [oldcell[2][0], oldcell[2][1], new_z]]
        self.set_cell(newcell)
        self.center(axis=2)
