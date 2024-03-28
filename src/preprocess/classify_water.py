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
import traceback

from lib.voxel import read_xyzv
from lib.pdb import get_coordinates_from_pdb, get_all_pdb_names
from lib.path import get_water_path, get_original_ligand_pocket_path, get_ghecom_ligand_pocket_path
from WaterClassifier.WaterClassifier import WaterClassifier
from WaterClassifier.ClassifyingRuleFactory import ClassifyingRuleFactory
from WaterClassifier.LigandPocketDefinerFactory import LigandPocketDefinerFactory

LIGAND_VOXEL_NUM = 6
LIGAND_POCKET_TYPE = "ghecom"
CLASSIFYING_RULE = "WaterClassifyingRuleCenter"
LIGAND_POCKET_DEFINER = "LigandPocketDefinerGhecom"
PATH_TYPE = f"LIGAND_POCKET_VOXEL_NUM_{LIGAND_VOXEL_NUM}" if LIGAND_POCKET_TYPE == "original" else "ghecom"
# PATH_TYPE = f"{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/"

def main():
    pdb_names = get_all_pdb_names()
    for pdb_name in pdb_names:
        pdb_name = '4b74'
        print(pdb_name)

        water_path = get_water_path(pdb_name)
        water_coordinates = get_coordinates_from_pdb(water_path)
        _, grid_dims, grid_origin = read_xyzv(pdb_name)

        paths = {
                "output_displaceable": os.path.join(f"/home/ito/research/data/{PATH_TYPE}/labeled_water/displaceable/", pdb_name,  f"pred_O_placed_{pdb_name}_3.0_test.pdb"),
                "output_non_displaceable": os.path.join(f"/home/ito/research/data/{PATH_TYPE}/labeled_water/non_displaceable/", pdb_name,  f"pred_O_placed_{pdb_name}_3.0_test.pdb"),
            }
        

        try:
            water_classifier = WaterClassifier(pdb_name, grid_dims, grid_origin)
            classifying_rule_factory = ClassifyingRuleFactory()
            ligand_pocket_definer_factory = LigandPocketDefinerFactory()
            ligand_pocket_definer = ligand_pocket_definer_factory.get_ligand_pocket_definer(LIGAND_POCKET_DEFINER, pdb_name, grid_dims, grid_origin, LIGAND_VOXEL_NUM)
            water_classifying_rule = classifying_rule_factory.get_rule(CLASSIFYING_RULE, pdb_name, grid_dims, grid_origin)
            water_classifier.define_ligand_pocket(ligand_pocket_definer)
            water_classifier.create_convert_dict(water_coordinates)
            displaceable_water_ids, non_displaceable_water_ids = water_classifier.get_classified_water_ids(water_coordinates, water_classifying_rule)
            water_classifier.save_classified_water_as_pdb(displaceable_water_ids=displaceable_water_ids,
                                                    non_displaceable_water_ids=non_displaceable_water_ids,
                                                    output_path_displaceable=paths["output_displaceable"],
                                                    output_path_non_displaceable=paths["output_non_displaceable"])
        except Exception as e:
            print(f"Error: {e}\n{traceback.format_exc()}")
            continue
        exit()
if __name__ == '__main__':
    main()