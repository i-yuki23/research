import sys
import numpy as np
sys.path.append("/home/ito/research/src/")
from lib.pdb import get_coordinates_from_pdb, get_all_pdb_names
from lib.path import get_ligand_path
from lib.helper import extract_points_within_threshold
from modules.read_ghecom import get_clusters_from_ghecom
from lib.voxel import extract_surroundings_voxel, read_xyzv, coordinate_to_voxel_index

# リガンドポケットを判定する
# 判定条件
# リガンド原子とクラスターの座標との距離がある閾値以下のgridをピック
SURROUNDING_VOXEL_NUM = 1


def extract_ligand_pocket_grids(ligand_coords ,ghecom_clusters, threshold):
    # 閾値以内にあるポケットの点を抽出
    extracted_points_list = {}
    for index, ghecom_coords in enumerate(ghecom_clusters):
        extracted_points = extract_points_within_threshold(ligand_coords, ghecom_coords, threshold)
        if extracted_points.size != 0:
            extracted_points_list[index] = extracted_points
    # print(extracted_points_list)
    return extracted_points_list

def get_dominant_cluster(extracted_points_list, ghecom_clusters):
    # 要素数が最も多いクラスターのインデックスを取得
    ligand_pocket_cluster_index = max(extracted_points_list.items(), key=len)[0]
    ligand_pocket_cluster_coords = ghecom_clusters[ligand_pocket_cluster_index]

    return ligand_pocket_cluster_coords

def get_ligand_pocket_form_ghecom(pdb_name, grid_dims, grid_origin, voxel_num): # originalがvoxel num で定義しているので合わせています
    threshold = voxel_num / 2  
    ligand_path = get_ligand_path(pdb_name)
    ligand_coords = get_coordinates_from_pdb(ligand_path) # リガンドの原子座標リスト（実際の座標に置き換える）
    ghecom_clusters = get_clusters_from_ghecom(pdb_name)  # ポケット領域を定義する点の座標リスト（実際の座標に置き換える）
    ligand_pocket_grids_list = extract_ligand_pocket_grids(ligand_coords, ghecom_clusters, threshold)
    if not ligand_pocket_grids_list:
        print("No points extracted.")
    ligand_pocket_grids = [item for sublist in ligand_pocket_grids_list.values() for item in sublist]  # listをまとめる（全グリットから抽出するとき）
    # ligand_pocket_grids = get_dominant_cluster(ligand_pocket_grids_list, ghecom_clusters)
    ligand_pocket_voxel_indices = coordinate_to_voxel_index(ligand_pocket_grids, grid_origin)
    ligand_pocket_voxel = extract_surroundings_voxel(ligand_pocket_voxel_indices, grid_dims, voxel_num=SURROUNDING_VOXEL_NUM)
    save_path = f'/home/ito/research/data/ghecom/ligand_pocket/{pdb_name}/ligand_pocket_from_grid.npy'
    # np.save(file=save_path, arr=ligand_pocket_voxel)
    return ligand_pocket_voxel

if __name__ == "__main__":
    pdb_names = get_all_pdb_names()
    for pdb_name in pdb_names:
        pdb_name = '4b74'
        get_ligand_pocket_form_ghecom(pdb_name)

        # exit()