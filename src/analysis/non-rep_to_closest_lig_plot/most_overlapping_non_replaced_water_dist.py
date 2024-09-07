

import sys
sys.path.append('../..')
from lib.pdb import get_coordinates_from_pdb, get_atoms_coords_for_each_atom_type, get_pdb_names_by_txt
from lib.path import get_ligand_path, get_predicted_non_replaced_water_path
from lib.helper import extract_points_within_threshold, calculate_closest_distance
import numpy as np
    
RADIUSES = {'C': 1.69984, 'N': 1.62500, 'O': 1.51369, 'S': 1.78180, 'B' : 1.92, 'F' : 1.47, 'P' : 1.80, 'I' : 1.98}
MAXIMUM_EMBEDDING_DISTANCE = {'C': 0.7564608700685973, 'O': 1.0939219279500616, 'S': 0.2330085575854186, 'N': 0.9927163533708043, 'B': 0.28465040269571507, 'P': 0.30781603, 'F': 0.6491095237687778, 'I': 0}
THRESHOLDS = {atom_type: RADIUSES[atom_type] + RADIUSES['O'] - MAXIMUM_EMBEDDING_DISTANCE[atom_type] for atom_type in RADIUSES.keys()}
# THRESHOLDS = {'C': 1.7006082598628054, 'N': 1.1532572932583913, 'O': 0.8395361440998768, 'S': 2.829472884829163, 'B': 2.8643891946085698, 'F': 1.6854709524624445, 'P': 2.69805794, 'I': 3.49369}

settings = {
        "DATA_TYPE" : 'gr',
        "DATA_VOXEL_NUM" : 10,
        "CLASSIFYING_RULE" : 'WaterClassifyingRuleEmbedding',
        "LIGAND_POCKET_DEFINER" : 'LigandPocketDefinerOriginal',
        "LIGAND_VOXEL_NUM" : 8,
        "MODEL_NAME" : 'ResNet',
        "is_augmented" : True
    }

def cal_dist_overlap_dist(water_coords, atom_coords, atom_type):
    dist = calculate_closest_distance(water_coords, atom_coords)
    return RADIUSES[atom_type] + RADIUSES['O'] - dist

def get_most_overlaping_non_replaced_water(water_coords, ligand_coords_for_each_atom_type):
    max_overlap = -1000
    max_overlap_list = []
    for water_coord in water_coords:
        for atom_type in ligand_coords_for_each_atom_type.keys():
            if len(ligand_coords_for_each_atom_type[atom_type]) != 0:
                max_overlap = max(max_overlap, cal_dist_overlap_dist([water_coord], ligand_coords_for_each_atom_type[atom_type], atom_type))
            else:
                max_overlap = None
            max_overlap_list.append(max_overlap)
    return max_overlap_list

def main():
    pdb_names = get_pdb_names_by_txt('../../../data/valid_test.txt')
    most_overlapping_non_replaced_water_list = []
    for pdb_name in pdb_names:
        predicted_non_replaced_water_path = get_predicted_non_replaced_water_path(pdb_name, settings['DATA_TYPE'], settings['DATA_VOXEL_NUM'], settings['LIGAND_POCKET_DEFINER'], settings['LIGAND_VOXEL_NUM'], settings['CLASSIFYING_RULE'], settings['MODEL_NAME'], settings['is_augmented'])
        predicted_non_replaced_water_coords = get_coordinates_from_pdb(predicted_non_replaced_water_path)
        if predicted_non_replaced_water_coords.size == 0:
            continue

        ligand_path = get_ligand_path(pdb_name)
        ligand_coords_for_each_atom_type = get_atoms_coords_for_each_atom_type(ligand_path)
        most_overlapping_non_replaced_water = get_most_overlaping_non_replaced_water(predicted_non_replaced_water_coords, ligand_coords_for_each_atom_type)
        most_overlapping_non_replaced_water_list.append(most_overlapping_non_replaced_water)
    most_overlapping_non_replaced_water_array = np.concatenate(most_overlapping_non_replaced_water_list, axis=0)
    np.save(f"{settings['CLASSIFYING_RULE']}_most_overlapping_non_replaced_water_array.npy", most_overlapping_non_replaced_water_array)
if __name__ == '__main__':
    main()