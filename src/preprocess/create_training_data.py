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
from Class.TrainingDataCreator.TrainingDataCreatorGr import TrainingDataCreatorGr

LIGNAD_POCKET_VOXEL_NUM = 9
CLASSIFYING_RULE = "WaterClassifyingRuleSurface"
LIGAND_POCKET_DEFINER = "LigandPocketDefinerOriginal"
DATA_VOXEL_NUM = 10

def main():
    training_data_creator = TrainingDataCreatorGr(pdb_name="4lkk", ligand_voxel_num=LIGNAD_POCKET_VOXEL_NUM, classifying_rule=CLASSIFYING_RULE, ligand_pocket_definer=LIGAND_POCKET_DEFINER, data_voxel_num=DATA_VOXEL_NUM)
    training_data_creator.save_training_data()

if __name__ == "__main__":
    main()