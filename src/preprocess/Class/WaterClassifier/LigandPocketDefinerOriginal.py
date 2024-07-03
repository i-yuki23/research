from modules.define_ligand_pocket import get_ligand_pocket
from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb
from lib.path import get_ligand_path, get_protein_path
from preprocess.Class.WaterClassifier.LigandPocketDefiner import LigandPocketDefiner

class LigandPocketDefinerOriginal(LigandPocketDefiner):

    def __init__(self, pdb_name, grid_dims, grid_origin, voxel_num):
        super().__init__(pdb_name, grid_dims, grid_origin, voxel_num)
        self.protein_coordinates = None
        self.protein_atomic_symbols = None
        self.ligand_coordinates = None


    def __load_data(self, pdb_name):
        self.protein_coordinates = get_coordinates_from_pdb(get_protein_path(pdb_name))
        self.protein_atomic_symbols = get_atomic_symbols_from_pdb(get_protein_path(pdb_name))
        self.ligand_coordinates = get_coordinates_from_pdb(get_ligand_path(pdb_name))

    def define_ligand_pocket(self):
        self.__load_data(self.pdb_name)
        
        return get_ligand_pocket(
            protein_coordinates=self.protein_coordinates,
            ligand_coordinates=self.ligand_coordinates,
            protein_atomic_symbols=self.protein_atomic_symbols,
            grid_dims=self.grid_dims,
            grid_origin=self.grid_origin,
            voxel_num=self.voxel_num
        )

