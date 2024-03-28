import numpy as np

class LigandPocketDefiner:
    def __init__(self, pdb_name, grid_dims, grid_origin, voxel_num):
        self.pdb_name = pdb_name
        self.grid_dims = grid_dims
        self.grid_origin = grid_origin
        self.voxel_num = voxel_num


    def define_ligand_pocket(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in subclass")