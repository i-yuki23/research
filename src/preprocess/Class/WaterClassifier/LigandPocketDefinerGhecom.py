from tmp.identify_ligand_pocket import get_ligand_pocket_form_ghecom
from Class.WaterClassifier.LigandPocketDefiner import LigandPocketDefiner

class LigandPocketDefinerGhecom(LigandPocketDefiner):

    def __init__(self, pdb_name, grid_dims, grid_origin, voxel_num):
        super().__init__(pdb_name, grid_dims, grid_origin, voxel_num)

    def define_ligand_pocket(self):         
        return get_ligand_pocket_form_ghecom(
            pdb_name=self.pdb_name,
            grid_dims=self.grid_dims,
            grid_origin=self.grid_origin,
            voxel_num=self.voxel_num
        )