import sys
sys.path.append('..')

import numpy as np
from lib.path import get_ligand_path
from lib.pdb import get_all_pdb_names, get_chains_coordinates, get_ligand_coordinates_from_pdb
from lib.voxel import coordinate_to_voxel_index, extract_surroundings_voxel, get_voxel_info
VOXEL_NUM = 4

all_protein = get_all_pdb_names()

data_dir = '/mnt/ito/pdbbind_raw/refined_set/'
valid_protein = []
for protein in all_protein:
    protein = '10gs'
    print(protein)
    multi_protein_file = data_dir + f'{protein}/{protein}_protein.pdb'
    protein_chains_coords_list = get_chains_coordinates(multi_protein_file)

    grid_dims, grid_origin = get_voxel_info(pdb_name=protein)
    protein_surroundings_list = []
    for chain_coords in protein_chains_coords_list:
        protein_indices = coordinate_to_voxel_index(chain_coords, grid_origin)
        protein_surroundings = extract_surroundings_voxel(protein_indices, grid_dims, VOXEL_NUM)
        protein_surroundings_list.append(protein_surroundings)
    summed_chains_coords = np.sum(protein_surroundings_list, axis=0)

    ligand_coordinates = get_ligand_coordinates_from_pdb(get_ligand_path(protein))
    ligand_indices = coordinate_to_voxel_index(ligand_coordinates, grid_origin)
    ligand_surroundings = extract_surroundings_voxel(ligand_indices, grid_dims, VOXEL_NUM)    
    common_voxel = (ligand_surroundings == 1) & (summed_chains_coords >= 2)
    if not np.any(common_voxel):
        valid_protein.append(protein)
    break
print(len(valid_protein))

with open('/home/ito/research/data/valid_protein.txt', 'w') as f:
    for protein in valid_protein:
        f.write(protein + '\n')

