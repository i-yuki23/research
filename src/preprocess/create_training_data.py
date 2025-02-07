import sys
sys.path.append('..')
from lib.path import get_training_data_dir, get_gr_path, get_protein_path, get_displaceable_water_path, get_non_displaceable_water_path
from lib.voxel import get_voxel_info
from lib.pdb import get_pdb_names_by_txt
from Class.TrainingDataCreator.TrainingDataCreatorGr import TrainingDataCreatorGr
from Class.TrainingDataCreator.TrainingDataCreatorGist import TrainingDataCreatorGist
from Class.TrainingDataCreator.TrainingDataCreatorProtein import TrainingDataCreatorProtein


LIGNAD_POCKET_VOXEL_NUM = 8
CLASSIFYING_RULE = "WaterClassifyingRuleEmbedding"
LIGAND_POCKET_DEFINER = "LigandPocketDefinerOriginal"
DATA_VOXEL_NUM = 10
DATA_TYPE = "gr"
PROTEIN_LIST_PATH = "/home/ito/research/data/all_valid_proteins"

def main():
    save_dir = get_training_data_dir(data_type=DATA_TYPE, data_voxel_num=DATA_VOXEL_NUM, classifying_rule=CLASSIFYING_RULE, ligand_pocket_definer=LIGAND_POCKET_DEFINER, ligand_voxel_num=LIGNAD_POCKET_VOXEL_NUM)
    pdb_names = get_pdb_names_by_txt(PROTEIN_LIST_PATH)

    for pdb_name in pdb_names:

        print(pdb_name)
        save_dir_dis = f"{save_dir}/displaceable/{pdb_name}/"
        save_dir_non_dis = f"{save_dir}/non_displaceable/{pdb_name}/"
        base_voxel_data_path = get_gr_path(pdb_name=pdb_name) 
        # base_voxel_data_path = get_protein_path(pdb_name=pdb_name)

        displaceable_water_path = get_displaceable_water_path(pdb_name=pdb_name, ligand_voxel_num=LIGNAD_POCKET_VOXEL_NUM, classifying_rule=CLASSIFYING_RULE, ligand_pocket_definer=LIGAND_POCKET_DEFINER)
        non_displaceable_water_path = get_non_displaceable_water_path(pdb_name=pdb_name, ligand_voxel_num=LIGNAD_POCKET_VOXEL_NUM, classifying_rule=CLASSIFYING_RULE, ligand_pocket_definer=LIGAND_POCKET_DEFINER)

        grid_dims, grid_origin = get_voxel_info(pdb_name)

        try:
            training_gr_data_creator = TrainingDataCreatorGr(data_voxel_num=DATA_VOXEL_NUM, grid_origin=grid_origin, grid_dims=grid_dims, save_dir_dis=save_dir_dis, save_dir_non_dis=save_dir_non_dis, base_voxel_data_path=base_voxel_data_path, displaceable_water_path=displaceable_water_path, non_displaceable_water_path=non_displaceable_water_path)
            training_gr_data_creator.save_training_data()
        except ValueError as e:
            print(f"Error processing {pdb_name}: {e}")
        except FileNotFoundError as e:
            print(f"Error processing {pdb_name}: {e}")
        except Exception as e:
            print(f"Error processing {pdb_name}: {e}")
            # exit(1)
        
        # try:
        #     training_gist_data_creator = TrainingDataCreatorGist(pdb_name=pdb_name, ligand_voxel_num=LIGNAD_POCKET_VOXEL_NUM, classifying_rule=CLASSIFYING_RULE, ligand_pocket_definer=LIGAND_POCKET_DEFINER, data_voxel_num=DATA_VOXEL_NUM)
        #     training_gist_data_creator.save_training_data()
        # except ValueError as e:
        #     print(f"Error processing {pdb_name}: {e}")
        # except FileNotFoundError as e:
        #     print(f"Error processing {pdb_name}: {e}")
        # except Exception as e:
        #     print(f"Error processing {pdb_name}: {e}")
        #     exit(1)

        # try:
        #     training_protein_data_creator = TrainingDataCreatorProtein(data_voxel_num=DATA_VOXEL_NUM, grid_origin=grid_origin, grid_dims=grid_dims, save_dir_dis=save_dir_dis, save_dir_non_dis=save_dir_non_dis, base_voxel_data_path=base_voxel_data_path, displaceable_water_path=displaceable_water_path, non_displaceable_water_path=non_displaceable_water_path)
        #     training_protein_data_creator.save_training_data()
        #     # exit()
        # except ValueError as e:
        #     print(f"Error processing {pdb_name}: {e}")
        # except FileNotFoundError as e:
        #     print(f"Error processing {pdb_name}: {e}")
        # except Exception as e:
        #     print(f"Error processing {pdb_name}: {e}")
        #     exit(1)
            

if __name__ == "__main__":
    main()