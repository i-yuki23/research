from identify_ligand_pocket import get_ligand_pocket_form_ghecom
from WaterClassifier.LigandPocketDefiner import LigandPocketDefiner

class LigandPocketDefinerGhecom(LigandPocketDefiner):

    def __init__(self, pdb_name, voxel_num):
        super().__init__(pdb_name, voxel_num)

    def define_ligand_pocket(self):         
        return get_ligand_pocket_form_ghecom(
            pdb_name=self.pdb_name,
            voxel_num=self.voxel_num
        )