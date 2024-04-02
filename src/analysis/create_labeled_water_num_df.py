import sys 
sys.path.append('..')

import pandas as pd
from lib.pdb import get_atom_id_from_pdb, get_all_pdb_names
from lib.path import get_displaceable_water_path, get_non_displaceable_water_path, get_analysis_results_labeled_water_path
from lib.helper import make_dir

def _count_water_number(water_pdb_path):
    ids = get_atom_id_from_pdb(water_pdb_path)
    return len(ids)

def calculate_water_numbers(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer):
    try:
        displaceable_water_num = _count_water_number(get_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer))
    except Exception as e:
        displaceable_water_num = None  # エラーがあった場合には None を設定

    try:
        non_displaceable_water_num = _count_water_number(get_non_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer))
    except Exception as e:
        non_displaceable_water_num = None  # エラーがあった場合には None を設定

    return displaceable_water_num, non_displaceable_water_num

def create_and_save_df(pdb_names, displaceable_water_nums, non_displaceable_water_nums, save_path):
    df = pd.DataFrame({
        'pdb_name': pdb_names,
        'displaceable_water_num': displaceable_water_nums,
        'non_displaceable_water_num': non_displaceable_water_nums
    })

    # 欠損値を扱う場合の一般的なアプローチとして、ここでは欠損値を0で置き換えます。
    # 別のアプローチが適切な場合は、それに応じて変更してください。
    df.fillna(0, inplace=True)
    make_dir(save_path)
    df.to_csv(save_path, index=False)

    # 基本統計量の計算
    # summary_stats = df.describe()

    # 合計の計算
    # total_displaceable = df['displaceable_water_num'].sum()
    # total_non_displaceable = df['non_displaceable_water_num'].sum()

def main():
    ligand_voxel_nums = [10, 9, 8, 6, 4]
    classifying_rules = ["WaterClassifyingRuleCenter", "WaterClassifyingRuleSurface"]
    ligand_pocket_definers = ["LigandPocketDefinerGhecom", "LigandPocketDefinerOriginal"]
    pdb_names = get_all_pdb_names()

    for ligand_voxel_num in ligand_voxel_nums:
        for classifying_rule in classifying_rules:
            for ligand_pocket_definer in ligand_pocket_definers:

                displaceable_water_nums = []
                non_displaceable_water_nums = []
                for pdb_name in pdb_names:
                    displaceable, non_displaceable = calculate_water_numbers(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)

                    displaceable_water_nums.append(displaceable)
                    non_displaceable_water_nums.append(non_displaceable)
                save_path = get_analysis_results_labeled_water_path(ligand_voxel_num, classifying_rule, ligand_pocket_definer)
                create_and_save_df(pdb_names, displaceable_water_nums, non_displaceable_water_nums, save_path)

            
if __name__ == '__main__':
    main()

