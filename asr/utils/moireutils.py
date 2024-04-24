from typing import Union
import numpy as np
from ase import Atoms


def get_shifts(uid_a: Union[str, None] = None,
               uid_b: Union[str, None] = None,
               database: str = '/home/niflheim2/steame/moire/utils/c2db.db',
               filename: str = 'shifts.json'):
    from ase.db import connect
    import json
    db = connect(database)
    get_a = db.get(uid=uid_a)
    get_b = db.get(uid=uid_b)
    dct = {
        'shift_v1': get_a.vbm_gw - get_a.vbm,
        'shift_c1': get_a.cbm_gw - get_a.cbm,
        'shift_v2': get_b.vbm_gw - get_b.vbm,
        'shift_c2': get_b.cbm_gw - get_b.cbm
    }
    with open(filename, 'w') as out:
        json.dump(dct, out)


def get_layers(atoms, swap: bool = False):
    '''divide structure in top and bottom layer,
       according to tags and atoms positions along z.

       If the starting bilayer is a 'draft', i.e. the
       two monolayers have not been vertically separated yet,
       it will only divide according to the tags.
    '''
    copy = atoms.copy()
    tags = copy.get_tags()
    unique = np.unique(tags)
    assert len(unique) == 2, \
        f'A bilayer can only have 2 different values for tags! You have {len(unique)}'

    l1 = copy[tags == unique[1]]
    l2 = copy[tags == unique[0]]
    if swap:
        return l2, l1
    return l1, l2


class Bilayer(Atoms):
    '''Adds some functionality to the Atoms class in order to
       make some operations for heterostructures easier.

       It requires that tags have been set by using two different integers
       in order to identify the atoms belonging to each layer.
    '''
    def __init__(self, atoms):
        super().__init__(atoms)
        self.top_layer, self.bottom_layer = get_layers(atoms)
        #self._atoms = self

    def copy(self):
        '''Returns a copy of itself.

        This is necessary because for some reason the copy() method
        inherited from the Atoms class fails (unexpected keyword argument 'cell')
        '''
        return Bilayer(Atoms(self).copy())
        #return self._atoms.copy()

    def get_total_thickness(self):
        zvals = self.positions[:, 2]
        return zvals.max() - zvals.min()

    def get_interlayer_distance(self):
        '''Returns interlayer distance,
           defined as the thickness of the vacuum
           region between the two layers
        '''
        ceiling = self.top_layer.positions[:, 2].min()
        floor = self.bottom_layer.positions[:, 2].max()
        distance = ceiling - floor
        return distance

    def set_interlayer_distance(self, distance: float):
        '''Sets interlayer distance according to the above definition'''
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
        sorter = np.argsort(self.positions[:, 2])
        old = self.copy()
        new = old[sorter]
        if order == 'descending':
            new = new[::-1]
        elif isinstance(order, list):
            new = old[order]
        self.__init__(new)

    def set_vacuum(self, value):
        '''Sets the total amount of vacuum between two adjacent cells
           along the z direction
        '''
        oldcell = self.cell
        thick = self.get_total_thickness()
        new_z = thick + value
        newcell = [oldcell[0],
                   oldcell[1],
                   [oldcell[2][0], oldcell[2][1], new_z]]
        self.set_cell(newcell)
        self.center(axis=2)
