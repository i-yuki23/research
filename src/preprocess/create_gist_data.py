# gistのデータをゲットしたい
# 水分子を中心として南北せるとかじゃなくて、水が存在しているところはpredでそれ以外は０にしたい
# ボクセル数がどんくらいになるかを確認→ボクセル数を合わせる
import sys
import os
sys.path.append('..')
import numpy as np
from lib.pdb import get_all_pdb_names
from lib.voxel import coordinate_to_voxel_index
from modules.get_gist_map_info import get_gist_map_info
from modules.get_labeled_water_coords import get_displaceable_water_coords, get_non_displaceable_water_coords
from modules.voxelizer import voxelizer_atom
from modules.extract_training_data_voxel import extract_training_data_voxel
from lib.path import get_training_data_path
from lib.helper import make_dir

WATER_CHANNEL_INDEX = 2
WATER_PRESENCE_THRESHOLD = 10**(-6)

LIGAND_VOXEL_NUM = 10
CLASSIFYING_RULE = "WaterClassifyingRuleCenter"
LIGAND_POCKET_DEFINER = "LigandPocketDefinerGhecom"
DATA_VOXEL_NUM = 10


def __get_gist_map_in_water_array(water_coords: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray, gist_map: np.ndarray) -> np.ndarray:
    gist_map_in_water_list = []
    for water_coord in water_coords:
        one_water_voxelized = voxelizer_atom(
            atomic_symbols=['O'],
            atom_coordinates=[water_coord],  # atom_coordinatesは2次元配列を受け取るから[]が必要
            grid_origin=grid_origin,
            grid_dims=grid_dims,
        )
        gist_map_in_water = np.where((one_water_voxelized[WATER_CHANNEL_INDEX] > WATER_PRESENCE_THRESHOLD), gist_map, 0)
        gist_map_in_water_list.append(gist_map_in_water)

    return np.array(gist_map_in_water_list)

def get_gist_data(pdb_name, gist_map, grid_dims, grid_origin, ligand_voxel_num, classifying_rule, ligand_pocket_definer):

    # GIST計算の結果をよりこみボクセルを取得
    # displaceable と non-displaceable の水分子の座標を取得
    # 水分子１っこをそれぞれボクセル化
    # 閾値10^-6でカットオフをかけ、ΔG_replaceをとってくる

    displaceable_water_coords = get_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    non_displaceable_water_coords = get_non_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    
    displaceable_gist_map_array = __get_gist_map_in_water_array(displaceable_water_coords, grid_dims, grid_origin, gist_map)
    non_displaceable_gist_map_array = __get_gist_map_in_water_array(non_displaceable_water_coords, grid_dims, grid_origin, gist_map)

    return displaceable_gist_map_array, non_displaceable_gist_map_array


def save_gist_training_data(pdb_name, gist_map, grid_dims, grid_origin, ligand_voxel_num, classifying_rule, ligand_pocket_definer, data_voxel_num) -> None:

    displaceable_gist_map_array, non_displaceable_gist_map_array = get_gist_data(pdb_name, gist_map, grid_dims, grid_origin, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    displaceable_water_coords = get_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    non_displaceable_water_coords = get_non_displaceable_water_coords(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)

    for displaceable_water_coord, displaceable_gist_map in zip(displaceable_water_coords, displaceable_gist_map_array):
        displaceable_gist_data = extract_training_data_voxel(water_coordinate=displaceable_water_coord, gr_or_gist=displaceable_gist_map, grid_origin=grid_origin, grid_dims=grid_dims, data_voxel_num=data_voxel_num)
        displaceable_save_path = get_training_data_path("gist", "displaceable", data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num, pdb_name)
        make_dir(displaceable_save_path)
        np.save(displaceable_save_path, displaceable_gist_data)
    
    for non_displaceable_water_coord, non_displaceable_gist_map in zip(non_displaceable_water_coords, non_displaceable_gist_map_array):
        non_displaceable_gist_data = extract_training_data_voxel(water_coordinate=non_displaceable_water_coord, gr_or_gist=non_displaceable_gist_map, grid_dims=grid_dims, grid_origin=grid_origin, data_voxel_num=data_voxel_num)
        non_displaceable_save_path = get_training_data_path("gist", "non_displaceable", data_voxel_num, classifying_rule, ligand_pocket_definer, ligand_voxel_num, pdb_name)
        make_dir(non_displaceable_save_path)
        np.save(non_displaceable_save_path, non_displaceable_gist_data)


def main():

    pdb_names = get_all_pdb_names()
    for pdb_name in pdb_names[9:]:
        print(pdb_name)

        try:
            gist_map, grid_dims, grid_origin = get_gist_map_info(pdb_name)
        except FileNotFoundError:
            continue

        try:
            save_gist_training_data(pdb_name, gist_map, grid_dims, grid_origin, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, DATA_VOXEL_NUM)
        except Exception as e:
            print(e)
            continue
        # exit()

# from multiprocessing import Process
if __name__ == '__main__':
    main()
    # process1 = Process(target=main)
    # process2 = Process(target=main)


    # process1.start()
    # process2.start()


    # process1.join()
    # process2.join()


