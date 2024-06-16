import sys
sys.path.append('../..')

from modules.define_ligand_pocket import get_ligand_pocket
from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb
from lib.path import get_test_ligand_path, get_test_protein_path, get_test_gr_path
from lib.voxel import get_voxel_info

class LigandPocketDefiner:

    def __init__(self, apo_name, holo_name, voxel_num):
        self.voxel_num = voxel_num
        self.grid_dims, self.grid_origin = get_voxel_info(dx_path=get_test_gr_path(apo_name))
        self.protein_coordinates = get_coordinates_from_pdb(get_test_protein_path(apo_name))
        self.protein_atomic_symbols = get_atomic_symbols_from_pdb(get_test_protein_path(apo_name))
        self.ligand_coordinates = get_coordinates_from_pdb(get_test_ligand_path(apo_name, holo_name), type="HETATM")

    def define_ligand_pocket(self):
        
        return get_ligand_pocket(
            protein_coordinates=self.protein_coordinates,
            ligand_coordinates=self.ligand_coordinates,
            protein_atomic_symbols=self.protein_atomic_symbols,
            grid_dims=self.grid_dims,
            grid_origin=self.grid_origin,
            voxel_num=self.voxel_num
        )

