from Class.LigandPocketDefiner import LigandPocketDefiner
from Class.LigandPocketWaterMoleculesExtractor import LigandPocketWaterMoleculesExtractor
from Class.VoxelDataExtractor import VoxelDataExtractor

class TestDataCreator:
    def __init__(self, ligand_pocket_definer, ligand_pocket_water_molecules_extractor, voxel_data_extractor):
        self.ligand_pocket_definer = ligand_pocket_definer
        self.ligand_pocket_water_molecules_extractor = ligand_pocket_water_molecules_extractor
        self.voxel_data_extractor = voxel_data_extractor

    def run(self):
        ligand_pockets = self.ligand_pocket_definer.define_ligand_pocket()
        
        ligand_pocket_water_coords, ligand_pocket_water_ids = self.ligand_pocket_water_molecules_extractor.get_water_coords_with_ids_inside_ligand_pocket(ligand_pockets)
        self.voxel_data_extractor.save_test_data(ligand_pocket_water_coords, ligand_pocket_water_ids)


ligand_pocket_definer = LigandPocketDefiner("2h14", "3smrA", 8)
ligand_pocket_water_molecules_extractor = LigandPocketWaterMoleculesExtractor("2h14")
voxel_data_extractor = VoxelDataExtractor("2h14", "3smrA", "/mnt/ito/test/2h14/2h14.dx", 10)


pipeline = TestDataCreator(
    ligand_pocket_definer=ligand_pocket_definer,
    ligand_pocket_water_molecules_extractor=ligand_pocket_water_molecules_extractor,
    voxel_data_extractor=voxel_data_extractor
)
pipeline.run()




