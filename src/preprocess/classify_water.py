"""
正解データのラベリングをする
水分子を中心とした周囲１０ボクセルの水和分布を取得し、それに対してラベリングを行う
その際にまず、水分子に対してラベリングを行う

正解データの条件
    リガンドポケット内あり、かつリガンドと重なる水分子→1(displaceable)
    リガンドポケット内にあり、かつリガンドと重ならない水分子→0(non-displaceable)

手順
1. placeventのpdbファイルから水分子の座標を取得
   → 座標をボクセル番号に変換、この際辞書でボクセル番号から座標を取得できるようにする
2. リガンドポケットのnpファイルをロード
3. リガンドをボクセル化
4. 水分子の中心がリガンドポケット内にあるかつ、リガンドと重なっているものを取得(重なっていないものを取得)
5. 抽出された水分子の中心のボクセル番号を取得
6. ボクセル番号から水分子の座標を取得
7. それぞれ分類した水分子をpdbファイルに書き込む
"""

import os
import sys
sys.path.append('..')
from lib.voxel import read_xyzv
from lib.pdb import get_coordinates_from_pdb, get_all_pdb_names
from lib.path import get_water_path, get_original_ligand_pocket_path, get_ghecom_ligand_pocket_path
from WaterClassifier.WaterClassifierWaterSurface import WaterClassifierWaterSurface

LIGAND_VOXEL_NUMS = [6, 4]
LIGAND_POCKET_TYPE = "ghecom"
# PATH_TYPE = f"LIGAND_POCKET_VOXEL_NUM_{LIGAND_VOXEL_NUM}" if LIGAND_POCKET_TYPE == "original" else "ghecom"

def main():
    pdb_names = get_all_pdb_names()
    for LIGAND_VOXEL_NUM in LIGAND_VOXEL_NUMS:
        PATH_TYPE = f"LIGAND_POCKET_VOXEL_NUM_{LIGAND_VOXEL_NUM}" if LIGAND_POCKET_TYPE == "original" else "ghecom"
        for pdb_name in pdb_names:
            # pdb_name = '4lkk'
            print(pdb_name)

            water_path = get_water_path(pdb_name)
            water_coordinates = get_coordinates_from_pdb(water_path)
            _, grid_dims, grid_origin = read_xyzv(pdb_name)

            paths = {
                    "output_displaceable": os.path.join(f"/home/ito/research/data/{PATH_TYPE}/labeled_water/displaceable/", pdb_name,  f"pred_O_placed_{pdb_name}_3.0_surface.pdb"),
                    "output_non_displaceable": os.path.join(f"/home/ito/research/data/{PATH_TYPE}/labeled_water/non_displaceable/", pdb_name,  f"pred_O_placed_{pdb_name}_3.0_surface.pdb"),
                }

            if LIGAND_POCKET_TYPE == "original":
                ligand_pocket_path = get_original_ligand_pocket_path(pdb_name, LIGAND_VOXEL_NUM)

            elif LIGAND_POCKET_TYPE == "ghecom":
                ligand_pocket_path = get_ghecom_ligand_pocket_path(pdb_name)

            try:
                water_classifier = WaterClassifierWaterSurface(pdb_name=pdb_name, grid_dims=grid_dims, grid_origin=grid_origin, ligand_pocket_path=ligand_pocket_path)
                water_classifier.load_ligand_and_pocket()
                water_classifier.create_convert_dict(water_coordinates)
                displaceable_water_ids, non_displaceable_water_ids = water_classifier.get_classified_water_ids(water_coordinates)
                water_classifier.save_classified_water_as_pdb(displaceable_water_ids=displaceable_water_ids,
                                                        non_displaceable_water_ids=non_displaceable_water_ids,
                                                        output_path_displaceable=paths["output_displaceable"],
                                                        output_path_non_displaceable=paths["output_non_displaceable"])
            except Exception as e:
                print(f"Error: {e}")
                continue
            # exit()
if __name__ == '__main__':
    main()