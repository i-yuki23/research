import numpy as np

class LigandPocketDefiner:
    def __init__(self, protein_path, ligand_path, grid_dims, grid_origin, voxel_num):
        self.protein_path = protein_path
        self.ligand_path = ligand_path
        self.grid_dims = grid_dims
        self.grid_origin = grid_origin
        self.voxel_num = voxel_num


    def define_ligand_pocket(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in subclass")