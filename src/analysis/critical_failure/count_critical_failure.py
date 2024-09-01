"""
致命的に間違っているタンパク質がどのくらいあるのかを調べる

- 致命的に間違っているタンパク質とは、
    - non-replace水分子(予測)の最近接リガンド原子の距離がある閾値以内のもの

手順
1. non-replace水分子の最近接リガンド原子を取得。
2. その距離を計算
3. 原子タイプを考慮して、閾値を設定
4. 閾値以内ならtrue, そうでないならfalseを返す
"""
import sys
sys.path.append('../..')
from lib.pdb import get_coordinates_from_pdb, get_atoms_coords_for_each_atom_type, get_pdb_names_by_txt
from lib.path import get_ligand_path, get_predicted_non_replaced_water_path
from lib.helper import extract_points_within_threshold
    
RADIUSES = {'C': 1.69984, 'N': 1.62500, 'O': 1.51369, 'S': 1.78180, 'B' : 1.92, 'F' : 1.47, 'P' : 1.80, 'I' : 1.98}
MAXIMUM_EMBEDDING_DISTANCE = {'C': 0.7564608700685973, 'O': 1.0939219279500616, 'S': 0.2330085575854186, 'N': 0.9927163533708043, 'B': 0.28465040269571507, 'P': 0.30781603, 'F': 0.6491095237687778, 'I': 0}
THRESHOLDS = {atom_type: RADIUSES[atom_type] + RADIUSES['O'] - MAXIMUM_EMBEDDING_DISTANCE[atom_type] for atom_type in RADIUSES.keys()}
# THRESHOLDS = {'C': 1.7006082598628054, 'N': 1.1532572932583913, 'O': 0.8395361440998768, 'S': 2.829472884829163, 'B': 2.8643891946085698, 'F': 1.6854709524624445, 'P': 2.69805794, 'I': 3.49369}

settings = {
        "DATA_TYPE" : 'gr',
        "DATA_VOXEL_NUM" : 10,
        "CLASSIFYING_RULE" : 'WaterClassifyingRuleSurface',
        "LIGAND_POCKET_DEFINER" : 'LigandPocketDefinerOriginal',
        "LIGAND_VOXEL_NUM" : 8,
        "MODEL_NAME" : 'ResNet',
        "is_augmented" : True
    }

def count_failed_water(pdb_name, model_settings):
    predicted_non_replaced_water_path = get_predicted_non_replaced_water_path(pdb_name, model_settings['DATA_TYPE'], model_settings['DATA_VOXEL_NUM'], model_settings['LIGAND_POCKET_DEFINER'], model_settings['LIGAND_VOXEL_NUM'], model_settings['CLASSIFYING_RULE'], model_settings['MODEL_NAME'], model_settings['is_augmented'])
    predicted_non_replaced_water_coords = get_coordinates_from_pdb(predicted_non_replaced_water_path)
    if predicted_non_replaced_water_coords.size == 0:
        return 0

    ligand_path = get_ligand_path(pdb_name)
    ligand_coords_for_each_atom_type = get_atoms_coords_for_each_atom_type(ligand_path)

    failed_water_num_total = 0
    for atom_type in ligand_coords_for_each_atom_type.keys():
        failed_water_coords = extract_points_within_threshold(ligand_coords_for_each_atom_type[atom_type], predicted_non_replaced_water_coords, THRESHOLDS[atom_type])
        failed_water_num_total += failed_water_coords.shape[0]
    # if failed_water_num_total > 0:
        # print(f"pdb_name: {pdb_name}, failed_water_num_total: {failed_water_num_total}")
    return failed_water_num_total

pdb_names = get_pdb_names_by_txt('../../../data/valid_test.txt')
failed_protein_num = 0
total_failed_water_num = 0
succeed_protein = []
for pdb_name in pdb_names:
    failed_water_num = count_failed_water(pdb_name, settings)
    if failed_water_num > 0:
        failed_protein_num += 1
    else:
        succeed_protein.append(pdb_name)

    total_failed_water_num += failed_water_num


print(failed_protein_num, '/', len(pdb_names))
print(total_failed_water_num)

with open(f'../../../data/{settings["CLASSIFYING_RULE"]}_success_protein_test.txt', 'w') as f:
    for pdb_name in succeed_protein:
        f.write(pdb_name + '\n')