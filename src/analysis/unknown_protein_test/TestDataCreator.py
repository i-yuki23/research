from Class.LigandPocketDefiner import LigandPocketDefiner

class TestDataCreator:
    def __init__(self, ligand_pocket_definer, ligand_pocket_water_molecules_extractor, voxel_data_extractor):
        self.ligand_pocket_definer = ligand_pocket_definer
        self.ligand_pocket_water_molecules_extractor = ligand_pocket_water_molecules_extractor
        self.voxel_data_extractor = voxel_data_extractor

    def run(self, protein_structure):
        ligand_pockets = self.ligand_pocket_definer.define_pockets(protein_structure)
        
        ligand_pocket_water_molecules = self.ligand_pocket_water_molecules_extractor.extract(ligand_pockets)
        
        self.voxel_data_extractor.extract_and_save(ligand_pocket_water_molecules)


ligand_pocket_definer = LigandPocketDefiner("2h14", "3smrA", 8)
ligand_pocket_water_molecules_extractor = LigandPocketWaterMoleculesExtractor()
voxel_data_extractor = VoxelDataExtractor()

hydration_pipeline = TestDataCreator(
    ligand_pocket_definer=ligand_pocket_definer,
    ligand_pocket_water_molecules_extractor=ligand_pocket_water_molecules_extractor,
    voxel_data_extractor=voxel_data_extractor
)


