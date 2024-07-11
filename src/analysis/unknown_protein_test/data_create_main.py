from TestDataCreator import TestDataCreator
from Class.LigandPocketDefiner import LigandPocketDefiner
from Class.LigandPocketWaterMoleculesExtractor import LigandPocketWaterMoleculesExtractor
from Class.VoxelDataExtractor import VoxelDataExtractor
from lib.path import get_test_gr_path

def main():
    protein_name = 'CDK2'
    apo_name = "1pw2"
    holo_name = '1pxi'
    ligand_voxel_num = 8
    data_voxel_num = 10
    base_voxel_path = get_test_gr_path(protein_name, apo_name)

    ligand_pocket_definer = LigandPocketDefiner(protein_name=protein_name, apo_name=apo_name, holo_name=holo_name, voxel_num=ligand_voxel_num)
    ligand_pocket_water_molecules_extractor = LigandPocketWaterMoleculesExtractor(protein_name=protein_name, apo_name=apo_name)
    voxel_data_extractor = VoxelDataExtractor(protein_name= protein_name, apo_name=apo_name, holo_name=holo_name, base_voxel_path=base_voxel_path, data_voxel_num=data_voxel_num)

    test_data_creator = TestDataCreator(
        ligand_pocket_definer=ligand_pocket_definer,
        ligand_pocket_water_molecules_extractor=ligand_pocket_water_molecules_extractor,
        voxel_data_extractor=voxel_data_extractor
    )

    test_data_creator.run()

if __name__ == '__main__':
    main()