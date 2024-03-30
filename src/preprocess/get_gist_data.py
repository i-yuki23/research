# gistのデータをゲットしたい
# 水分子を中心として南北せるとかじゃなくて、水が存在しているところはpredでそれ以外は０にしたい
# ボクセル数がどんくらいになるかを確認→ボクセル数を合わせる
import sys
sys.path.append('..')
import numpy as np
from lib.pdb import get_all_pdb_names
from lib.voxel import coordinate_to_voxel_index
from modules.get_gist_map_info import get_gist_map_info
from modules.get_labeled_water_coords import get_displaceable_water_coords, get_non_displaceable_water_coords
from modules.voxelizer import voxelizer_atom

WATER_CHANNEL_INDEX = 2
WATER_PRESENCE_THRESHOLD = 10**(-6)

LIGAND_VOXEL_NUM = 10
CLASSIFYING_RULE = "WaterClassifyingRuleCenter"
LIGAND_POCKET_DEFINER = "LigandPocketDefinerGhecom"


def __get_gist_map_in_water_array(water_coords: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray, gist_map: np.ndarray) -> np.ndarray:
    gist_map_in_water_list = []
    for one_water_coords in water_coords:
        one_water_voxelized = voxelizer_atom(
            atomic_symbols=['O'],
            atom_coordinates=[one_water_coords],  # atom_coordinatesは2次元配列を受け取るから[]が必要
            grid_origin=grid_origin,
            grid_dims=grid_dims,
        )
        gist_map_in_water = np.where((one_water_voxelized[WATER_CHANNEL_INDEX] > WATER_PRESENCE_THRESHOLD), gist_map, 0)
        gist_map_in_water_list.append(gist_map_in_water)

    return np.array(gist_map_in_water_list)

def get_gist_data(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer):

    # GIST計算の結果をよりこみボクセルを取得
    # displaceable と non-displaceable の水分子の座標を取得
    # 水分子１っこをそれぞれボクセル化
    # 閾値10^-6でカットオフをかけ、ΔG_replaceをとってくる

    gist_map, grid_dims, grid_origin = get_gist_map_info(pdb_name)

    displaceable_water_coords = get_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    non_displaceable_water_coords = get_non_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    
    displaceable_gist_map_array = __get_gist_map_in_water_array(displaceable_water_coords, grid_dims, grid_origin, gist_map)
    non_displaceable_gist_map_array = __get_gist_map_in_water_array(non_displaceable_water_coords, grid_dims, grid_origin, gist_map)

    return displaceable_gist_map_array, non_displaceable_gist_map_array

def get_labeled_water_voxel_indices(pdb_name, grid_origin, ligand_voxel_num, classifying_rule, ligand_pocket_definer):

    displaceable_water_coords = get_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    non_displaceable_water_coords = get_non_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)

    displaceable_water_voxel_indices = np.array([coordinate_to_voxel_index(coordinate, grid_origin) for coordinate in displaceable_water_coords])
    non_displaceable_water_voxel_indices = np.array([coordinate_to_voxel_index(coordinate, grid_origin) for coordinate in non_displaceable_water_coords])

    return displaceable_water_voxel_indices, non_displaceable_water_voxel_indices

def create_gist_training_data(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer, data_voxel_num) -> None:
    displaceable_gist_map_array, non_displaceable_gist_map_array = get_gist_data(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    displaceable_water_voxel_indices, non_displaceable_water_voxel_indices = get_labeled_water_voxel_indices(pdb_name, grid_origin, ligand_voxel_num, classifying_rule, ligand_pocket_definer)

    return

def main():

    pdb_names = get_all_pdb_names()
    for pdb_name in pdb_names:
        print(pdb_name)
        displaceable_gist_map_array, non_displaceable_gist_map_array = get_gist_data(pdb_name, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER)
        print(displaceable_gist_map_array)
        exit()

if __name__ == '__main__':
    main()