import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import os
from modules.voxelizer import voxelizer_atom
from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb, get_all_pdb_names
from lib.path import get_ligand_path, get_protein_path
from lib.voxel import read_xyzv, coordinate_to_voxel_index
from lib.voxel import extract_surroundings_voxel

# Constants for readability
PROTEIN_PRESENT_THRESHOLD = 1-np.exp(-1)

def is_protein_present(protein_voxel, threshold=PROTEIN_PRESENT_THRESHOLD):
    return np.any(protein_voxel > threshold, axis=0)


def get_ligand_pocket(protein_coordinates, ligand_coordinates, protein_atomic_symbols, grid_dims, grid_origin, voxel_num):
    """
    Identifies the ligand pocket in a protein structure.
    
    Parameters:
    - base_path: str, base path to the data directory
    - PDB: str, PDB ID of the structure
    - voxel_num: int, size of the voxel area to consider around each atom
    
    Returns:
    - np.ndarray, voxel grid representation of the ligand pocket
    """

    protein_indices = coordinate_to_voxel_index(protein_coordinates, grid_origin)
    ligand_indices = coordinate_to_voxel_index(ligand_coordinates, grid_origin)

    protein_surroundings = extract_surroundings_voxel(protein_indices, grid_dims, voxel_num)
    ligand_surroundings = extract_surroundings_voxel(ligand_indices, grid_dims, voxel_num)
    # print(protein_surroundings.shape, ligand_surroundings.shape)
    protein_voxel = voxelizer_atom(
        protein_atomic_symbols, 
        protein_coordinates,
        grid_origin,
        grid_dims,
        half_length_index_cutoff=5,
        length_voxel=0.5
    )

    protein_presence = is_protein_present(protein_voxel)
    ligand_pocket = np.where((protein_surroundings == ligand_surroundings) & 
                             (protein_surroundings == 1) &
                             (~protein_presence)
                             ,1, 0)
    return ligand_pocket


def main():
    pdb_names = get_all_pdb_names()

    for pdb_name in pdb_names:
        # pdb_name = '4lkk'
        print(f"Processing {pdb_name}...")
        try:
            protein_coordinates = get_coordinates_from_pdb(get_protein_path(pdb_name))
            protein_atomic_symbols = get_atomic_symbols_from_pdb(get_protein_path(pdb_name))
            ligand_coordinates = get_coordinates_from_pdb(get_ligand_path(pdb_name))
            _, grid_dims, grid_origin = read_xyzv(pdb_name)
            # print(protein_coordinates.shape, ligand_coordinates.shape, atomic_symbols.shape, grid_dims, grid_origin)
            ligand_pocket = get_ligand_pocket(
                protein_coordinates=protein_coordinates,
                ligand_coordinates=ligand_coordinates,
                protein_atomic_symbols=protein_atomic_symbols,
                grid_dims=grid_dims,
                grid_origin=grid_origin
            )
            np.save(f"/home/ito/research/data/ligand_pocket/{pdb_name}/VOXEL_NUM_{VOXEL_NUM}.npy", ligand_pocket)
            # exit(0)

        except FileNotFoundError as e:
            print(e)
            print(f"Failed to identify ligand pocket for {pdb_name}.")
            continue

if __name__ == '__main__':
    main()
