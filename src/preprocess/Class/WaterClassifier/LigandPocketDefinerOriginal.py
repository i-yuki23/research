from modules.define_ligand_pocket import get_ligand_pocket
from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb
from lib.path import get_ligand_path, get_protein_path
from preprocess.Class.WaterClassifier.LigandPocketDefiner import LigandPocketDefiner

class LigandPocketDefinerOriginal(LigandPocketDefiner):

    def __init__(self, protein_path, ligand_path, grid_dims, grid_origin, voxel_num):
        super().__init__(protein_path, ligand_path, grid_dims, grid_origin, voxel_num)

    def __load_data(self):
        protein_coordinates = get_coordinates_from_pdb(self.protein_path)
        protein_atomic_symbols = get_atomic_symbols_from_pdb(self.protein_path)
        ligand_coordinates = get_coordinates_from_pdb(self.ligand_path)

        return protein_coordinates, protein_atomic_symbols, ligand_coordinates

    def define_ligand_pocket(self):
        protein_coordinates, protein_atomic_symbols, ligand_coordinates = self.__load_data()
        
        return get_ligand_pocket(
            protein_coordinates=protein_coordinates,
            ligand_coordinates=ligand_coordinates,
            protein_atomic_symbols=protein_atomic_symbols,
            grid_dims=self.grid_dims,
            grid_origin=self.grid_origin,
            voxel_num=self.voxel_num
        )

