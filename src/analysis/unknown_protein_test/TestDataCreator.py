class TestDataCreator:
    def __init__(self, ligand_pocket_definer, ligand_pocket_water_molecules_extractor, voxel_data_extractor):
        self.ligand_pocket_definer = ligand_pocket_definer
        self.ligand_pocket_water_molecules_extractor = ligand_pocket_water_molecules_extractor
        self.voxel_data_extractor = voxel_data_extractor

    def run(self):
        ligand_pockets = self.ligand_pocket_definer.define_ligand_pocket()
        
        ligand_pocket_water_coords, ligand_pocket_water_ids = self.ligand_pocket_water_molecules_extractor.get_water_coords_with_ids_inside_ligand_pocket(ligand_pockets)
        self.voxel_data_extractor.save_test_data(ligand_pocket_water_coords, ligand_pocket_water_ids)




