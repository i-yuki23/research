def get_ligand_path(pdb_name):
    return f"/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{pdb_name}/{pdb_name}_ligand.pdb"

def get_water_path(pdb_name):
    return f"/mnt/ito/3DRISM220308/{pdb_name}/pred_O_placed_{pdb_name}_3.0.pdb"

def get_original_ligand_pocket_path(pdb_name, ligand_voxel_num):
    return f"/home/ito/research/data/ligand_pocket/{pdb_name}/VOXEL_NUM_{ligand_voxel_num}.npy"

def get_ghecom_ligand_pocket_path(pdb_name):
    return f"/home/ito/research/data/ghecom/ligand_pocket/{pdb_name}/ligand_pocket.npy"

def get_protein_path(pdb_name):
    return f"/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{pdb_name}/{pdb_name}_min.pdb"

def get_gr_path(pdb_name):
    return f"/mnt/ito/3DRISM220308/{pdb_name}/{pdb_name}_min.dx"

def get_gist_path(pdb_name):
    return f"/mnt/ito/3DRISM220308/{pdb_name}/{pdb_name}_gist-pred.dx"

def get_nas_data_dir():
    return "/mnt/dandan/3drism/dir_3DRISM_20181213_155211/"

def get_home_data_dir():
    return "/home/ito/research/data/"

def get_xyzv_path(pdb_name):
    return f"/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{pdb_name}/analysis.O.1.xyzv"

def get_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer):
    return f"/home/ito/research/data/labeled_water/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/displaceable/{pdb_name}/pred_O_placed_{pdb_name}_3.0.pdb"

def get_non_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer):
    return f"/home/ito/research/data/labeled_water/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/non_displaceable/{pdb_name}/pred_O_placed_{pdb_name}_3.0.pdb"
    
def get_training_data_path(data_type: str, water_type: str, data_voxel_num:int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int, pdb_name: str, water_id: int):
    return f"/mnt/ito/data/training_data/{data_type}/data_voxel_num_{data_voxel_num}/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/{water_type}/{pdb_name}/water_id_{water_id}.npy"

def get_training_data_dir(data_type: str, data_voxel_num:int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int):
    return f"/mnt/ito/data/training_data/{data_type}/data_voxel_num_{data_voxel_num}/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/"

def get_analysis_results_labeled_water_path(ligand_voxel_num, classifying_rule, ligand_pocket_definer):
    return f"/home/ito/research/data/analysis_results/labeled_water/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/labeled_water_numbers.csv"

def get_ghecom_path(pdb_name):
    return f"/home/ito/research/data/ghecom/clusters/{pdb_name}/{pdb_name}_ghecom.pdb"

def get_hist_path(data_type: str, data_voxel_num:int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int, model_name: str, trainer_name:str):
    return f"/home/ito/research/src/history/{data_type}/data_voxel_num_{data_voxel_num}/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/{model_name}/{trainer_name}/training_history.json"

def get_predicted_labeled_water_path(data_type, pdb_name, data_voxel_num, ligand_voxel_num, classifying_rule, ligand_pocket_definer, model_name, predicted_type, is_augmented):
    if is_augmented:
        return f"/home/ito/research/data/analysis_results/predicted_labeled_water/{data_type}/data_voxel_num_{data_voxel_num}/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/{model_name}/aug_train/{pdb_name}/{predicted_type}/pred_O_placed_{pdb_name}_3.0.pdb"
    return f"/home/ito/research/data/analysis_results/predicted_labeled_water/{data_type}/data_voxel_num_{data_voxel_num}/{ligand_pocket_definer}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule}/{model_name}/normal_train/{pdb_name}/{predicted_type}/pred_O_placed_{pdb_name}_3.0.pdb"

def get_crystal_water_path(pdb_name):
    return f"/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{pdb_name}/HOH.pdb"

def get_test_gr_path(protein_name, apo_name):
    return f"/mnt/ito/test/{protein_name}/apo/{apo_name}.dx"

def get_test_protein_path(protein_name, apo_name):
    return f"/mnt/ito/test/{protein_name}/apo/{apo_name}.pdb"

def get_test_ligand_path(protein_name, holo_name):
    return f"/mnt/ito/test/{protein_name}/holo/{holo_name}/{holo_name}_ligand.pdb"

def get_test_water_path(protein_name, apo_name):
    return f"/mnt/ito/test/{protein_name}/apo/pred_O_placed_{apo_name}_3.0.pdb"

def get_checkpoints_dir(DATA_TYPE, DATA_VOXEL_NUM, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, MODEL_NAME, is_augmented):
    if is_augmented:
        return f'/home/ito/research/src/checkpoints/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/aug_train/'
    return f'/home/ito/research/src/checkpoints/valid/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/normal_train/'

def get_predicted_non_replaced_water_path(pdb_name, DATA_TYPE, DATA_VOXEL_NUM, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, MODEL_NAME, is_augmented):
    if is_augmented:
        return f"/home/ito/research/data/analysis_results/predicted_labeled_water/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/aug_train/{pdb_name}/non_displaceable/pred_O_placed_{pdb_name}_3.0.pdb"
    return f"/home/ito/research/data/analysis_results/predicted_labeled_water/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/normal_train/{pdb_name}/non_displaceable/pred_O_placed_{pdb_name}_3.0.pdb"

def get_predicted_replaced_water_path(pdb_name, DATA_TYPE, DATA_VOXEL_NUM, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, MODEL_NAME, is_augmented):
    if is_augmented:
        return f"/home/ito/research/data/analysis_results/predicted_labeled_water/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/aug_train/{pdb_name}/displaceable/pred_O_placed_{pdb_name}_3.0.pdb"
    return f"/home/ito/research/data/analysis_results/predicted_labeled_water/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/normal_train/{pdb_name}/displaceable/pred_O_placed_{pdb_name}_3.0.pdb"