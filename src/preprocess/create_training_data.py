import sys
sys.path.append('..')
from lib.pdb import get_all_pdb_names
from Class.TrainingDataCreator.TrainingDataCreatorGr import TrainingDataCreatorGr
from Class.TrainingDataCreator.TrainingDataCreatorGist import TrainingDataCreatorGist
from Class.TrainingDataCreator.TrainingDataCreatorProtein import TrainingDataCreatorProtein


LIGNAD_POCKET_VOXEL_NUM = 8
CLASSIFYING_RULE = "WaterClassifyingRuleSurface"
LIGAND_POCKET_DEFINER = "LigandPocketDefinerOriginal"
DATA_VOXEL_NUM = 10

def main():
    pdb_names = get_all_pdb_names()

    for pdb_name in pdb_names:
        # pdb_name = '4lkk'
        print(pdb_name)

        # try:
        #     training_gr_data_creator = TrainingDataCreatorGr(pdb_name=pdb_name, ligand_voxel_num=LIGNAD_POCKET_VOXEL_NUM, classifying_rule=CLASSIFYING_RULE, ligand_pocket_definer=LIGAND_POCKET_DEFINER, data_voxel_num=DATA_VOXEL_NUM)
        #     training_gr_data_creator.save_training_data()
        #     exit()
        # except ValueError as e:
        #     print(f"Error processing {pdb_name}: {e}")
        # except FileNotFoundError as e:
        #     print(f"Error processing {pdb_name}: {e}")
        # except Exception as e:
        #     print(f"Error processing {pdb_name}: {e}")
        #     exit(1)
        
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

        try:
            training_protein_data_creator = TrainingDataCreatorProtein(pdb_name=pdb_name, ligand_voxel_num=LIGNAD_POCKET_VOXEL_NUM, classifying_rule=CLASSIFYING_RULE, ligand_pocket_definer=LIGAND_POCKET_DEFINER, data_voxel_num=DATA_VOXEL_NUM)
            training_protein_data_creator.save_training_data()
            # exit()
        except ValueError as e:
            print(f"Error processing {pdb_name}: {e}")
        except FileNotFoundError as e:
            print(f"Error processing {pdb_name}: {e}")
        except Exception as e:
            print(f"Error processing {pdb_name}: {e}")
            exit(1)
            

if __name__ == "__main__":
    main()