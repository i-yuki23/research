import os
import glob
from lib.path import get_training_data_dir

def get_water_nums_by_pdb_list(pdb_names: list, data_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int) -> list:
    water_nums = []
    for pdb_name in pdb_names:
        water_nums.append(get_water_num_by_pdb(pdb_name, data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num))

def get_displaceable_water_nums_by_pdb_list(pdb_names: list, data_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int) -> list:
    water_nums = []
    for pdb_name in pdb_names:
        water_nums.append(get_displaceable_water_num_by_pdb(pdb_name, data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num))
    return water_nums

def get_non_displaceable_water_nums_by_pdb_list(pdb_names: list, data_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int) -> list:
    water_nums = []
    for pdb_name in pdb_names:
        water_nums.append(get_non_displaceable_water_num_by_pdb(pdb_name, data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num))
    return water_nums

def get_water_num_by_pdb(pdb_name: str, data_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int):
    return get_displaceable_water_num_by_pdb(pdb_name, data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num) + get_non_displaceable_water_num_by_pdb(pdb_name, data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num)

def get_displaceable_water_num_by_pdb(pdb_name: str, data_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int):
    water_dir = get_training_data_dir("gr", data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num) + f"displaceable/{pdb_name}/"
    return len(glob.glob(os.path.join(water_dir, '*.npy')))

def get_non_displaceable_water_num_by_pdb(pdb_name: str, data_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str, ligand_voxel_num: int):
    water_dir = get_training_data_dir("gr", data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num) + f"non_displaceable/{pdb_name}/"
    return len(glob.glob(os.path.join(water_dir, '*.npy')))

