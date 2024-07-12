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

from lib.voxel import get_voxel_info
from lib.pdb import get_coordinates_from_pdb, get_all_pdb_names
from lib.path import get_water_path
from lib.helper import make_dir
from Class.WaterClassifier.WaterClassifier import WaterClassifier
from Class.WaterClassifier.ClassifyingRuleFactory import ClassifyingRuleFactory
from Class.WaterClassifier.LigandPocketDefinerFactory import LigandPocketDefinerFactory
DATA_DIR = "/home/ito/research/data/labeled_water/"

def classify_water(pdb_name, ligand_voxel_num, classifying_rule_name, ligand_pocket_definer_name):
    
    PATH_TYPE = f"{ligand_pocket_definer_name}/ligand_pocket_voxel_num_{ligand_voxel_num}/{classifying_rule_name}/"
        # pdb_name = '4b74'
    print(pdb_name)

    paths = {
            "output_displaceable": os.path.join(DATA_DIR, PATH_TYPE, "displaceable/", pdb_name,  f"pred_O_placed_{pdb_name}_3.0.pdb"),
            "output_non_displaceable": os.path.join(DATA_DIR, PATH_TYPE, "non_displaceable/", pdb_name,  f"pred_O_placed_{pdb_name}_3.0.pdb"),
        }
    
    for path in paths.values():
        make_dir(path)

    water_path = get_water_path(pdb_name)
    water_coordinates = get_coordinates_from_pdb(water_path)
    grid_dims, grid_origin = get_voxel_info(pdb_name)

    water_classifier = WaterClassifier(pdb_name, grid_dims, grid_origin)
    
    classifying_rule_factory = ClassifyingRuleFactory()
    ligand_pocket_definer_factory = LigandPocketDefinerFactory()
    ligand_pocket_definer = ligand_pocket_definer_factory.get_ligand_pocket_definer(ligand_pocket_definer_name, pdb_name, grid_dims, grid_origin, ligand_voxel_num)
    water_classifying_rule = classifying_rule_factory.get_rule(classifying_rule_name, pdb_name, grid_dims, grid_origin)
    water_classifier.define_ligand_pocket(ligand_pocket_definer)
    water_classifier.create_convert_dict(water_coordinates)
    displaceable_water_ids, non_displaceable_water_ids = water_classifier.get_classified_water_ids(water_coordinates, water_classifying_rule)
    # print(displaceable_water_ids, non_displaceable_water_ids)
    water_classifier.save_classified_water_as_pdb(displaceable_water_ids=displaceable_water_ids,
                                            non_displaceable_water_ids=non_displaceable_water_ids,
                                            output_path_displaceable=paths["output_displaceable"],
                                            output_path_non_displaceable=paths["output_non_displaceable"])


def main():
    ligand_voxel_nums = [8]
    classifying_rule_names = ["WaterClassifyingRuleEmbedding"]
    ligand_pocket_definer_names = ["LigandPocketDefinerOriginal"]
    pdb_names = get_all_pdb_names()

    for ligand_voxel_num in ligand_voxel_nums:
        for classifying_rule_name in classifying_rule_names:
            for ligand_pocket_definer_name in ligand_pocket_definer_names:
                for pdb_name in pdb_names:

                    # pdb_name = '2d3u'
                    # ligand_pocket_definer_name = 'LigandPocketDefinerOriginal'
                    # classifying_rule_name = 'WaterClassifyingRuleSurface'
                    # ligand_voxel_num = 10

                    print(f"ligand_voxel_num: {ligand_voxel_num}, classifying_rule: {classifying_rule_name}, ligand_pocket_definer: {ligand_pocket_definer_name}")
                    try:
                        classify_water(pdb_name, ligand_voxel_num, classifying_rule_name, ligand_pocket_definer_name)
                    except ValueError as e:
                        # 特定のエラーを捕捉して処理。ここではエラーメッセージを出力するだけ
                        print(f"Error processing {pdb_name}: {e}")
                        continue
                    except Exception as e:
                        # 予期しない他のエラーを捕捉して処理
                        print(f"Unexpected error processing {pdb_name}: {e}")
if __name__ == "__main__":
    main()