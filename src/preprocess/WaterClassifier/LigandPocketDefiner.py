import numpy as np

class LigandPocketDefiner:
    def __init__(self, pdb_name, voxel_num):
        self.pdb_name = pdb_name
        self.voxel_num = voxel_num


    def define_ligand_pocket(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in subclass")