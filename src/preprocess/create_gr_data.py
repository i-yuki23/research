"""
水分子の周囲10ボクセルのgrのボクセルデータを取ってくる

手順
1. pdbから水分子の座標を取ってくる
2. 座標をボクセルのインデックスに変換
3. 周囲10ボクセルのgrを取得
"""
import sys
sys.path.append('..')
import os
import numpy as np
from lib.pdb import get_atoms_coordinates_from_pdb
from lib.voxel import coordinate_to_voxel_index
from lib.dx import read_dx
GR_VOXEL_NUM = 10
LIGNAD_POCKET_VOXEL_NUM = 6

def main():
    dir_path = "/mnt/dandan/3drism/dir_3DRISM_20181213_155211"
    pdb_names = [pdb_name for pdb_name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, pdb_name))]
    for pdb_name in pdb_names:
        # pdb_name = '4lkk'
        print(f"Processing {pdb_name}...")
        try:
            paths = {
                "displaceable_water" : f"/home/ito/research/data/LIGAND_POCKET_VOXEL_NUM_{LIGNAD_POCKET_VOXEL_NUM}/labeled_water/displaceable/{pdb_name}/pred_O_placed_{pdb_name}_3.0.pdb",
                "non_displaceable_water" : f"/home/ito/research/data/LIGAND_POCKET_VOXEL_NUM_{LIGNAD_POCKET_VOXEL_NUM}/labeled_water/non_displaceable/{pdb_name}/pred_O_placed_{pdb_name}_3.0.pdb",
                "gr" : f"/mnt/ito/3DRISM220308/{pdb_name}/{pdb_name}_min.dx"
            }
            displaceable_waters_coordinates = get_atoms_coordinates_from_pdb(paths["displaceable_water"])
            non_displaceable_waters_coordinates = get_atoms_coordinates_from_pdb(paths["non_displaceable_water"])
            gr, grid_dims, grid_origin = read_dx(paths["gr"])

            def save_gr_dataset(waters_coordinates, gr, grid_origin, grid_dims, water_type):
                water_num_count = 0
                for water_coordinates in waters_coordinates:
                    gr_data = get_gr_data(water_coordinates, gr, grid_origin, grid_dims)
                    np.save(f"/home/ito/research/data/LIGAND_POCKET_VOXEL_NUM_{LIGNAD_POCKET_VOXEL_NUM}/gr_data/GR_VOXEL_NUM_{GR_VOXEL_NUM}/{water_type}/{pdb_name}/water_no_{water_num_count}.npy", gr_data)
                    water_num_count += 1

            save_gr_dataset(displaceable_waters_coordinates, gr, grid_origin, grid_dims, "displaceable")
            save_gr_dataset(non_displaceable_waters_coordinates, gr, grid_origin, grid_dims, "non_displaceable")
            # exit(0)

        except FileNotFoundError as e:
            print(e)
            print(f"Failed to identify water gr data for {pdb_name}.")
            continue

if __name__ == "__main__":
    main()