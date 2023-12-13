from asr.core import command, option, AtomsFile, ASRResult, prepare_result
from typing import List
from ase import Atoms
import numpy as np
import spglib
from ase.io import read, write
import os
from ase.geometry import wrap_positions
from ase.geometry import get_distances
from asr.core import write_json, read_json
from asr.database.rmsd import get_rmsd
from functools import cached_property

class StackingError(ValueError):
    ''' only 2D materials can be stacked. Rasie error if 0D or 3D'''
    pass


class OriginError(ValueError):
    ''' There must be an atom at the in-plane origin'''
    pass


class LatticeSymmetries:
    def __init__(self, mlatoms, spglib_tol=0.1, remove_unphysical=True):

        self.spglib_tol = spglib_tol
        self.remove_unphysical= remove_unphysical
        self.mlatoms = mlatoms


    @cached_property
    def mlSym(self):
        return spglib.get_symmetry(self.mlatoms, symprec=self.spglib_tol)


    @cached_property
    def braveSym(self):
        ''' symmetries of the brave lattice 
            if asked to remove unphysical symmetries remove det=-1'''
        pure_cell = Atoms('C', positions=[[0, 0, 0]])
        pure_cell.set_cell(self.mlatoms.get_cell())
        # We want to be more strict on the Brave lattice symmetries not to generate too many transformations
        #symC = spglib.get_symmetry(pure_cell, symprec=self.spglib_tol)
        symC = spglib.get_symmetry(pure_cell, symprec=0.01)
        if not self.remove_unphysical:
            return symC
        else:
            pure_rotations =    [s for s, t in zip(symC['rotations'], symC['translations']) if np.linalg.det(s)!=-1]
            pure_translations = [t for s, t in zip(symC['rotations'], symC['translations']) if np.linalg.det(s)!=-1]
        return {'rotations': pure_rotations, 'translations':pure_translations}


    @cached_property
    def brave_not_monolayer_sym(self):
        '''We will remov the symmetries of monolayer except the identity transformation '''
        pure_rots = self.braveSym['rotations']
        pure_trans = self.braveSym['translations']
        ml_rots = self.mlSym['rotations']
        rots  = [arr              for iarr, arr in enumerate(pure_rots) if not any(np.array_equal(arr, arr2) for arr2 in ml_rots)]
        trans = [pure_trans[iarr] for iarr, arr in enumerate(pure_rots) if not any(np.array_equal(arr, arr2) for arr2 in ml_rots)]
        rotations = [np.array([[1,0,0],[0,1,0],[0,0,1]])]
        translations = [np.array([0,0,0])]
        rotations.extend(rots)
        translations.extend(trans)
        return {'rotations': rotations, 'translations':translations}


    @cached_property
    def brave_not_monolayer_not_flip_sym(self):
        '''We will treat flipping of layers differently so we remove flipping transformations'''
        res = self.brave_not_monolayer_sym
        rots_, trans_ = res['rotations'], res['translations']         
        rots = [r          for ii, r in enumerate(rots_) if r[2,2]!=-1]
        trans = [trans_[ii] for ii, r in enumerate(rots_) if r[2,2]!=-1]
        return {'rotations': rots, 'translations':trans}


    @cached_property
    def ml_needs_flip(self):
        '''If the monolayer changes with flipping we want to know and build bilayers accordingly
        We want to do this only when there is a transformation that flips by rotating around the first unit vector'''
        #return all([s[2,2]==1 for s in self.mlSym['rotations']]) and any([mat in self.braveSym['rotations'] if np.linalg.det(mat[0:2, 0:2])==-1 and mat[0,0]==1 and mat[1,0]==0 and mat[2,2]==-1])

        for mat in self.braveSym['rotations']:
            block = mat[0:2, 0:2]
            if np.linalg.det(block)==-1 and mat[0,0]==1 and mat[1,0]==0 and mat[2,2]==-1:
                return all([s[2,2]==1 for s in self.mlSym['rotations']])
        else:
            return False


    @cached_property
    def cell_type(self):
        ''' this is related to lattice symmetry but used for displacement vectors'''
        cell = self.mlatoms.get_cell()
        cell[:, 2] = 0.0
        cell[2, :] = 0.0
        assert cell.rank == 2, cell
        lat = cell.get_bravais_lattice()
        name = lat.name
        if name == 'HEX2D':
            name = 'hexagonal'
        elif name == 'RECT':
            name = 'rectangular'
        elif name == 'SQR':
            name = 'square'
        elif name == 'CRECT':
            name = 'centered'
        else:
            name = 'oblique'
        return name


class InplaneTranslations:
    def __init__(self, mlatoms, cell_type, flatten_tol=0.2, hexagonal_thirds=False, bridge_points=False):
        self.mlatoms = mlatoms
        self.cell_type = cell_type
        self.flatten_tol = flatten_tol
        self.hexagonal_thirds = hexagonal_thirds
        self.bridge_points = bridge_points

    def flatten(self, atoms=None):
        '''returns a list of positons'''
        if atoms is None: 
            atoms=self.atoms
            
        flats = []
        for atom in atoms:
            pos = atom.position[:2]
            if any([np.allclose(x, pos, self.flatten_tol) for x in flats]):
                continue
            else:
                flats.append(pos)
        return flats


    def cell_specific_stacks(self, atoms=None):
        if atoms is None:
            atoms=self.mlatoms

        final_transforms = []
        positions = atoms.get_positions()
        a, b, c = atoms.cell.lengths()

        def append_helper(x, y, atoms):
            # The helper is to bring the atom in the origin to the for example middle of the cell
            # The for loop is to bring all the atoms to the middle of the cell
            for atom in atoms:
                x_ = x-atom.position[0]
                y_ = y-atom.position[1]
                final_transforms.append(np.array([x_, y_]))

        # This is there to generate the middle of the cell and vectors
        if self.cell_type in ['oblique', 'rectangular', 'square', 'centered', 'hexagonal']:
            vec1 = atoms.cell[0]
            vec2 = atoms.cell[1]

            new_vec = vec1/2
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = vec2/2
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = (vec1+vec2)/2
            append_helper(new_vec[0], new_vec[1], atoms)

        if self.cell_type == 'hexagonal':
            if len(positions) != 1:
                # we want to avoid cases where the first 2 atoms are on top of each other

                for pos in positions:
                    if np.linalg.norm(positions[0, 0:2]-pos[0:2])>0.1:
                        second_atom = pos
                        break

                x = second_atom[0]
                y = second_atom[1]
                append_helper(x, y, atoms)

                x = 2 * second_atom[0]
                y = 2 * second_atom[1]
                append_helper(x, y, atoms)

        # Thos condition is set to create the 1/3, 2/3 displacements.
        # We have deactivated it because it depends on the choice of cell vectors and it generates many more compared to the above method
        if self.hexagonal_thirds and self.cell_type in ['hexagonal']:
            vec1 = atoms.cell[0]
            vec2 = atoms.cell[1]

            new_vec = 1*vec1/3 + 2*vec2/3
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = 2*vec1/3 + 1*vec2/3
            append_helper(new_vec[0], new_vec[1], atoms)
            new_vec = 1*vec1/3 + 1*vec2/3
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = 2*vec1/3 + 2*vec2/3
            append_helper(new_vec[0], new_vec[1], atoms)

        return final_transforms


    @staticmethod
    def vector_in_list_pbc(cell, vec1, veclist):
        dist = []
        for vec2 in veclist:
            _temp = Atoms('C2', positions=[vec1, vec2])
            _temp.set_cell(cell)
            pos = _temp.positions
            d1, d2= get_distances([pos[0]],[pos[1]],cell=_temp.cell, pbc=[True, True, False])
            dist.append(np.max(d2.flatten()))
        return any([d<0.3 for d in dist])


    def pairwise_displacements(self, atoms2, atoms1=None):
        if atoms1 is None:
            atoms1=self.mlatoms.copy()

        bot_flatten = self.flatten(atoms1)
        top_flatten = self.flatten(atoms2)

        # We start with AA and want to avoid regenrating same displacements
        displacements = [[0,0,0]]

        #put each atom of top layer on one atom in the bottom layer
        for pos1 in bot_flatten:
            for pos2 in top_flatten:
                move = pos1 - pos2
                move = [move[0],move[1],0]
                move = wrap_positions([move], atoms1.cell, pbc=True, center=(0.5, 0.5, 0.5), pretty_translation=False, eps=1e-07)[0]
                if not self.vector_in_list_pbc(cell=atoms1.cell, vec1=move, veclist=displacements):
                    displacements.append(move)
         
        #cell specific displacements should be applied on each atom at the origin
        for t in self.cell_specific_stacks(atoms=atoms2.copy()):
            ts = [t[0],t[1],0]
            ts = wrap_positions([ts], atoms1.cell, pbc=True, center=(0.5, 0.5, 0.5), pretty_translation=False, eps=1e-07)[0]
            if not self.vector_in_list_pbc(cell=atoms1.cell, vec1=ts, veclist=displacements):
                displacements.append(ts)

        
        # Now we want to generate displacement vectors in the middle of 2 atoms
        # we do it after previous steps to give it lower priority and remove these rather than previous ones 
        if self.bridge_points:
            for pos1 in bot_flatten:
                for pos2 in top_flatten:
                    move = (pos1 - pos2)/2
                    move = [move[0],move[1],0]
                    move = wrap_positions([[move[0],move[1],0]], atoms1.cell, pbc=True, center=(0.5, 0.5, 0.5), pretty_translation=False, eps=1e-07)[0]
                    if not self.vector_in_list_pbc(cell=atoms1.cell, vec1=move, veclist=displacements):
                        displacements.append(move)
               
        return displacements



class BilayerMatcher:
    def __init__(self, atoms_list, auxs_list, transform_layer, brave_symmetries, matcher_tol=0.3, use_rmsd=False):
        self.use_rmsd = use_rmsd   
        self.matcher_tol = matcher_tol
        self.atoms1 = atoms_list[0]
        self.atoms2 = atoms_list[1]
        self.atoms_list = atoms_list
        self.auxs_list = auxs_list
        self.transform_layer = transform_layer
        self.brave_symmetries = brave_symmetries

 
    @classmethod
    def atoms_distance_pbc(cls, cell, pos_1, pos_2):
        from ase.geometry import get_distances

        _temp = Atoms('C2', positions=[pos_1, pos_2])
        _temp.set_cell(cell)
        pos = _temp.positions
        d1, d2= get_distances([pos[0]],[pos[1]],cell=_temp.cell, pbc=[True, True, False])
        return  np.max(d2.flatten())  #np.max(d2.flatten())<10e-4


    @classmethod
    def atoms_equal(cls, atoms1, atoms2, matcher_tol, use_rmsd=False):
        ''' Is there exactly one atom in atoms2 that is at same position and has same atomic number? '''
        # We don't want to use rmsd this is for testing our comparator against rmsd
        if use_rmsd:
            return (get_rmsd(atoms1.copy(), atoms2.copy()) or 1) < matcher_tol

        # The following compares structures atom by atom
        poss1, poss2 = atoms1.get_positions(), atoms2.get_positions()
        numbers1, numbers2 = atoms1.get_atomic_numbers(), atoms2.get_atomic_numbers()

        for (pos1, n1) in zip(poss1, numbers1):
            dist = []
            count = 0
            for (pos2, n2) in zip(poss2, numbers2):
                if n1 != n2: continue
                dist.append(cls.atoms_distance_pbc(atoms1.cell, pos1, pos2))

            if len(dist) == 0: return False

            if min(dist)>matcher_tol:
                return False
            else:
                count += 1
                if count >= 2:
                    raise ValueError('Two atoms are on top of each other')

        return True


    def bilayers_similarity_family(self):
        '''Here for each bilayer we generate family members that are rotated with respect to it'''
        mats=self.atoms_list
        auxs=self.auxs_list
 
        family = []
        for ii, (mat, aux) in enumerate(zip(mats, auxs)):
            nml = int(len(mat)//2)
            members = []
            # Here we transform the whole bilayer with the symmetry of the brave lattice
            for bsym in self.brave_symmetries['rotations']:
                bot = mat[0:nml]
                top = mat[nml::]
                new_bot = self.transform_layer(bot, bsym, np.array([0,0,0]))
                new_top = self.transform_layer(top, bsym, np.array([0,0,0]))
                new_mat = new_bot+new_top
                new_mat.wrap(pbc=[1, 1, 1])
                # we want to make sure the first atom remain at the same place even in z-direction
                # this is needed because the first atom is not in the origin in z-direction
                ref_pos_old = mat.positions[0]
                ref_pos_new = new_mat.positions[0] if new_mat.positions[0,2]<new_mat.positions[nml,2] else new_mat.positions[nml]
                new_positions = [pos - ref_pos_new + ref_pos_old for pos in new_mat.positions ]
                new_mat.set_positions(new_positions)
                members.append({"new_mat":new_mat, "sym_trans":bsym, "original_mat":mat})

            family.append(members)    
        return mats, auxs, family


    def unique_materials(self):
        mats, auxs, family = self.bilayers_similarity_family()        

        unique_mats = [mats[0]]
        unique_auxs = [auxs[0]]
        unique_mats_equivalents = {}
        for ii, (mat, aux, submat) in enumerate(zip(mats, auxs, family)):

            new_bilayer = True
            for ui, x in enumerate(unique_mats):
                if ui not in unique_mats_equivalents.keys():
                    unique_mats_equivalents[ui]=[]
           
                for temp in submat:
                    if self.atoms_equal(atoms1=temp["new_mat"], atoms2=x, matcher_tol=self.matcher_tol, use_rmsd=self.use_rmsd):
                        new_bilayer = False
                        unique_mats_equivalents[ui].append({"original_mat":temp["original_mat"], "sym_trans":temp["sym_trans"]})
                    
            if new_bilayer:
                unique_mats.append(mat)
                unique_auxs.append(aux)
       
        return unique_mats, unique_auxs


class BuildBilayer(LatticeSymmetries):
    def __init__(self, mlatoms, mlfolder, spglib_tol, remove_unphysical, vacuum, matcher_tol, use_rmsd=False, hund_rule=False, bridge_points=False, prefix='II-'):
        super().__init__(mlatoms, spglib_tol, remove_unphysical)
        self.mlfolder = mlfolder
        self.bot = self.mlatoms if prefix[0]=='I' else self.flip_monolayer() 
        self.top = self.mlatoms if prefix[1]=='I' else self.flip_monolayer() 
        self.matcher_tol = matcher_tol
        self.use_rmsd = use_rmsd
        self.prefix = prefix
        self.vacuum = vacuum
        self.hund_rule = hund_rule
        self.bridge_points = bridge_points

    @cached_property
    def interlayer(self):
        """
        For the bilayer prototype we set the interlayer distance
        (cloest atoms distance) to be 6. This is hard coded.It does not
        have consequences as we do zscan to find the correct interlayer 
        But this value has to be fixed here for when we want to compare 
        bilayers to find the unique ones
        """
        width = max(self.top.positions[:,2])-min(self.top.positions[:,2])
        return 6 + width

    @classmethod
    def transform_layer(cls, atoms, U_cc, t_c):
        ''' gets a monolayer and returns transformed monolayer'''
        rotated_atoms = atoms.copy()
        # Calculate rotated and translated atoms
        spos_ac = rotated_atoms.get_scaled_positions()
        spos_ac = np.dot(spos_ac, U_cc.T) + t_c
        # Move atoms
        rotated_atoms.set_scaled_positions(spos_ac)
        # Wrap atoms outside of unit cell back
        rotated_atoms.wrap(pbc=[1, 1, 1])
        return rotated_atoms


    def translation(self, x, y, z, rotated, base):
        '''Combine rotated with base by translation.
        x, y, z are cartesian coordinates.'''
        stacked = base.copy()
        rotated = rotated.copy()
        rotated.translate([x, y, z])
        stacked += rotated
        # we adjust the vacuum here for when comparing to find unique bilayers
        stacked = self.adjust_bilayer_vacuum(vacuum=self.vacuum, bilayer=stacked)
        stacked.wrap()
        return stacked

    
    @classmethod
    def build_bilayer_from_files(blfoder=None, mlfolder=None, ignore_interlayer=False, interlayer=5):
        ''' You can use this function to build bilayers inside the folder that has needed files inside it
            if the required files are not available it will raise error
            you can use this without having interlayer distance'''
        if blfolder is None:
            blfolder = '.'
        if mlfolder is None:
            mlfolder = f'{blfolder}/..'

        if not os.path.isfile(f'{blfolder}/transformdata.json') or not os.path.isfile(f'{blfolder}/translation.json'):
            raise FileNotFoundError("Required files not available: 'transformdata.json' or 'translation.json'")
        if not ignore_interlayer and not os.path.isfile(f'{blfolder}/results-asr.zscan.json'):
            raise FileNotFoundError("You need to provide zscan result file or set ignore_interlayer=True")

        transform = read_json(f'{blfolder}/transformdata.json')
        translation = read_json(f'{blfolder}/translation.json')['translation_vector']
        Ucc = transform('rotation')
        tc = transform('translation')
        bot_flipped = transform('Bottom_layer_Flipped')
        top_flipped = transform('Top_layer_Flipped')
        bot = read(f'{mlfolder}/structure_flipped.json') if bot_flipped else read(f'{mlfolder}/structure.json')
        top = read(f'{mlfolder}/structure_flipped.json') if top_flipped else read(f'{mlfolder}/structure.json')
        rotated_top = cls.transform_layer(top, U_cc=Ucc, t_c=tc)
        bilayer = cls.translation(x=translation[0], y=translation[1], z=interlayer, rotated=rotated_top, base=bot)
        return bilayer


    def flip_monolayer(self):
        ''' saves and returns the flipped monolayer
            we also save the transformation that is used to flip the c2db monolayer'''
        if not self.ml_needs_flip:
            return None

        brave_rots = self.brave_not_monolayer_sym['rotations']
        for sym in brave_rots:
            if sym[2,2]==-1 and sym[0,0]==1 and sym[1,0]==0:
               dct = {'transformation': sym}
               write_json(f'{self.mlfolder}/flip_transformation.json', dct)
               flipping_sym = sym
               break
        flipped_monolayer = self.transform_layer(atoms=self.mlatoms, U_cc=flipping_sym, t_c=np.array([0,0,0]) )
        flipped_monolayer.write(f'{self.mlfolder}/structure_flipped.json')
        return flipped_monolayer
      

    def get_transformed_atoms(self, atoms):
        ''' here we apply symmetry transformations on the top layer
            It might be that two symmetry transformed top layers are equivalent so we should keep unique ones
            This filtering for unique top layers is the reason we dont merge this in the build bilayer function'''
        transformations=self.brave_not_monolayer_not_flip_sym

        transformed_tops = []
        transforms = []
        for U_cc, t_c in zip(transformations["rotations"], transformations["translations"]):
            rotated_atoms = self.transform_layer(atoms, U_cc, t_c)
            transformed_tops.append(rotated_atoms)
            transforms.append((U_cc, t_c))

        #we check the monolayers to be different
        monolayer_matcher = BilayerMatcher.atoms_equal
        unique_tops = [transformed_tops[0]]
        unique_transforms = [transforms[0]]
        for top, trans in zip(transformed_tops, transforms):
            for x in unique_tops:
                # For matching monolayers never use rmsd because it finds all monolayers the same
                if not monolayer_matcher(atoms1=top, atoms2=x, matcher_tol=self.matcher_tol, use_rmsd=False):
                    unique_tops.append(top)
                    unique_transforms.append(trans)

        print(">>>>>>>>>>>>>>> Removing same transformed monolayer: ", f"{len(transformed_tops)} >> {len(unique_tops)}")
        return unique_tops, unique_transforms
    

    def build_bilayers(self):
        #we want each layer to have magnetic moments of the monolayer
        self.set_initial_magmoms()

        rotated_tops, transforms = self.get_transformed_atoms(atoms=self.top)

        bottom_translations = InplaneTranslations(mlatoms=self.bot.copy(),
                                                  cell_type=self.cell_type,
                                                  flatten_tol=0.1,
                                                  hexagonal_thirds=False,
                                                  bridge_points=self.bridge_points)

        bilayers = []
        toplayer_transformations = []
        translations = []
        for toplayer, (U_cc, t_c) in zip(rotated_tops, transforms):
            displacements = bottom_translations.pairwise_displacements(atoms2=toplayer.copy())
            for disp in displacements:
                bilayer = self.translation(disp[0], disp[1], self.interlayer, toplayer, self.bot)

                bilayers.append(bilayer)
                toplayer_transformations.append((U_cc, t_c))
                translations.append(disp)

        auxs = list(zip(translations, toplayer_transformations))
        return bilayers, auxs

    
    def bilayer_folder_name(self, formula, nlayers, U_cc, t_c):
        def pretty_float(arr):
            f1 = round(arr[0], 2)
            s1 = "0" if np.allclose(f1, 0.0) else str(f1)
            f2 = round(arr[1], 2)
            s2 = "0" if np.allclose(f2, 0.0) else str(f2)
            return f'{s1}_{s2}'

        return f"{self.prefix}-{formula}-{nlayers}-{U_cc[0, 0]}_{U_cc[0, 1]}_{U_cc[1, 0]}_{U_cc[1, 1]}-{pretty_float(t_c)}"
        

    @staticmethod
    def adjust_bilayer_vacuum(vacuum, bilayer, layer1_indices=[], layer2_indices=[]):
        new_bilayer = bilayer.copy()
        layer1 = bilayer[::int(len(bilayer)//2)] if layer1_indices == [] else bilayer[layer1_indices]
        layer2 = bilayer[int(len(bilayer)//2)::] if layer2_indices == [] else bilayer[layer2_indices]           

        bilayer_zmin = min(bilayer.positions[:,2])
        bilayer_zmax = max(bilayer.positions[:,2])
        bilayer_width = bilayer_zmax - bilayer_zmin
        #length of the third cell vector
        zvec = vacuum + bilayer_width
      
        new_positions = new_bilayer.positions
        new_positions[:,2] += (-bilayer_zmin+vacuum/2)
        new_bilayer.set_positions(new_positions)
        new_bilayer.cell[2,2] *= (zvec/bilayer.cell[2,2])
        return new_bilayer


    def set_initial_magmoms(self):
        magmoms = []

        # If you come with a monolayer that you don't have magnetic information for it start with hund's law
        if self.hund_rule:
            TM3d_list = {'V': 1, 'Cr': 3, 'Mn': 5,'Fe': 4, 'Co': 3,'Ni': 2, 'Cu': 1}
            magmoms = [TM3d_list[atom.symbol] if atom.symbol in TM3d_list.keys() else 0 for atom in self.mlatoms]

        # set the magmoms of the relaxed monolayer for the initial values of the bilayer
        elif "magmom" in read_json(f"{self.mlfolder}/structure_initial.json")[1]:
            magmoms = read_json(f"{self.mlfolder}/structure_initial.json")[1]["magmoms"]
  
        if magmoms!=[]:
            self.top.set_initial_magnetic_moments(magmoms)
            self.bot.set_initial_magnetic_moments(magmoms)

    
    @staticmethod
    def set_original_folder(folder, key, value):
        '''We want to save the original bilayer 
           We will track the changes due to slidestability later
           There is a setinfo recipe but it does not accept folder address
        '''
        from pathlib import Path
        infofile = Path(f'./{folder}/info.json')
        if infofile.is_file():
            info = read_json(infofile)
        else:
            info = {}

        info[key] = value
        write_json(infofile, info)



    def make_bilayer_folders(self):
        bilayers, auxs = self.build_bilayers() 
        print('>>>>>>>>>>>>>>> Number of bilayers created: ', len(bilayers))

        matcher = BilayerMatcher(bilayers, auxs, self.transform_layer, brave_symmetries=self.braveSym, matcher_tol=self.matcher_tol, use_rmsd=self.use_rmsd)
        unique_bilayers, unique_auxs = matcher.unique_materials()
        print('>>>>>>>>>>>>>>> Number of unique bilayers: ', len(unique_bilayers))
        translations, syms = zip(*unique_auxs)

        # Here we have the bilayers we will now make the folders, and put the files inside them 
        names = []
        for transl, tform, proto in zip(translations, syms, unique_bilayers):
            # Unpack and transform data needed to construct bilayer name
            t = tform[1] + \
                self.top.cell.scaled_positions(np.array([transl[0], transl[1], 0.0]))
            name = self.bilayer_folder_name(self.top.get_chemical_formula(), 2, tform[0], t)
            names.append(name)

            # create bilayer folers and save the bilayer files inside them
            if not os.path.isdir(name):
                os.mkdir(name)

            self.set_original_folder(folder=name, key='original_bilayer_folder', value=name)

            start_bilayer = self.adjust_bilayer_vacuum(vacuum=self.vacuum, bilayer=proto) 
            start_bilayer.write(f'{name}/bilayerprototype.json')

            dct = {'translation_vector': transl}
            write_json(f'{name}/translation.json', dct)

            transform_data = {'rotation': tform[0],
                              'translation': tform[1]}
            if self.prefix[0]=='F':
                transform_data['Bottom_layer_Flipped']=True
            else:
                transform_data['Bottom_layer_Flipped']=False

            if self.prefix[1]=='F':
                transform_data['Top_layer_Flipped']=True
            else:
                transform_data['Top_layer_Flipped']=False

            write_json(f'{name}/transformdata.json', transform_data)

        return names


@prepare_result
class StackBilayerResult(ASRResult):
    folders: List[str]

    key_descriptions = dict(
        folders='Folders containing created bilayers')

@command(module='asr.stack_bilayer', 
         requires=['structure_initial.json'],
         creates = ['structure_adjusted.json'],
         returns=StackBilayerResult)

@option('-spgtol', '--spglib-tol', type=float,
        help='Position tolerance to determine the symmetries')
@option('-mtol', '--matcher-tol', type=float,
        help='Position tolerance for bilayer matcher')
@option('-ml', '--ml-folder', type=str,
        help='Monolayer folder to same the bilayer')
@option('-r', '--remove-unphysical', type=bool,
        help='Remove symmetry transformations of the brave lattice with det=-1')
@option('-rmsd', '--use-rmsd', type=bool,
        help='Use rmsd for comparing bilayers')
@option('-vac', '--vacuum', type=float,
        help='Vacuum for the perpenicular direction')
@option('--hund-rule', type=bool,
        help='Use hund rule for initial magnetic moments')
@option('--bridge-points', type=bool,
        help='When stacking consider the bridge sites as well')

def main(spglib_tol: float = 0.1,
         matcher_tol: float = 0.6,
         ml_folder: str= '.',
         remove_unphysical: bool=True,
         use_rmsd: bool=False,
         vacuum: float=15,
         hund_rule: bool=False,
         bridge_points: bool=False) -> StackBilayerResult: #ASRResult:

    atoms = read(f'{ml_folder}/structure_initial.json')

    if sum(atoms.pbc) != 2:
        raise StackingError('It is only possible to stack 2D materials')

    if np.linalg.norm(atoms.positions[0,0:2])>1e-3:
        #raise OriginError('There is no atom at the origin')
        print('>>>>>>>>>> WARNING: There was no atom at the origin')
        print(">>>>>>>>>> Moving the first atom to the origin")
        atoms.positions[:,0:2] -= atoms.positions[0,0:2]

    atoms = BuildBilayer.adjust_bilayer_vacuum(vacuum=vacuum, bilayer=atoms.copy(), layer1_indices=range(len(atoms)), layer2_indices=range(len(atoms)))
    atoms.write('structure_adjusted.json') 

    symmetries = LatticeSymmetries(mlatoms=atoms.copy(),  
                                   spglib_tol=spglib_tol, 
                                   remove_unphysical=remove_unphysical)

    build_bilayer_inputs = [atoms.copy(), ml_folder, spglib_tol, remove_unphysical, vacuum,  matcher_tol, use_rmsd, hund_rule, bridge_points]
    bilayer_folders_II = BuildBilayer(*build_bilayer_inputs, prefix='II')
    names = bilayer_folders_II.make_bilayer_folders()

    if symmetries.ml_needs_flip:
        print('>>>>>>>>>>>>>>>', 'The monolayer will be flipped')
        bilayer_folders_IF = BuildBilayer(*build_bilayer_inputs, prefix='IF')
        names += bilayer_folders_IF.make_bilayer_folders()
        bilayer_folders_FI = BuildBilayer(*build_bilayer_inputs, prefix='FI')
        names += bilayer_folders_FI.make_bilayer_folders()

    return StackBilayerResult.fromdata(folders=names)


if __name__ == '__main__':
    main.cli()
