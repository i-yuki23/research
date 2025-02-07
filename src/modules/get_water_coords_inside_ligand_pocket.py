import sys 
sys.path.append('..')

from typing import Tuple
import numpy as np
from lib.voxel import coordinate_to_voxel_index, get_voxel_info
from lib.pdb import get_coordinates_from_pdb, filter_atoms_and_create_new_pdb

def create_convert_dict(water_coordinates: np.ndarray, water_ids: np.ndarray, grid_origin: np.ndarray) -> None:
    water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)
    water_index_to_coordinate = {tuple(water_voxel_index[i]): water_coordinates[i] for i in range(water_voxel_index.shape[0])}
    water_coordinate_to_id = {tuple(water_coordinate): int(water_id) for water_coordinate, water_id in zip(water_coordinates, water_ids)}
    return water_index_to_coordinate, water_coordinate_to_id

def get_voxelized_water_center(water_coordinates: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> np.ndarray:
    water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)
    print(water_voxel_index)
    voxelized_water_center = np.zeros(grid_dims)
    for index in water_voxel_index:
        if (
            0 <= index[0] < grid_dims[0] and
            0 <= index[1] < grid_dims[1] and
            0 <= index[2] < grid_dims[2]
        ):
            voxelized_water_center[index[0], index[1], index[2]] = 1
        else:
            # 範囲外のインデックスを無視
            continue
    return voxelized_water_center
    

def convert_voxel_to_water_coordinates(water_voxelized: np.ndarray, water_index_to_coordinate: dict) -> np.ndarray:
    water_index_tmp = np.where(water_voxelized == 1)
    water_index = []
    for i in zip(water_index_tmp[0], water_index_tmp[1], water_index_tmp[2]):
        water_index.append([i[0], i[1], i[2]])
    print(water_index_tmp)
    water_coordinates = []
    for index in water_index:
        water_coordinates.append(water_index_to_coordinate[tuple(index)])
    return np.vstack(water_coordinates)

def get_water_coordinates_inside_ligand_pocket(water_coordinates: np.ndarray, water_ids: np.ndarray, ligand_pocket: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> np.ndarray:
    voxelized_water_center = get_voxelized_water_center(water_coordinates, grid_dims, grid_origin)
    voxelized_water_center_inside_ligand_pocket = np.where((voxelized_water_center == 1) & (ligand_pocket == 1), 1, 0)
    water_index_to_coordinate, _ = create_convert_dict(water_coordinates, water_ids, grid_origin)
    water_coordinates_inside_ligand_pocket = convert_voxel_to_water_coordinates(voxelized_water_center_inside_ligand_pocket, water_index_to_coordinate)

    return water_coordinates_inside_ligand_pocket

def save_water_coordinates_inside_ligand_pocket(input_pdb_path: str, output_pdb_path: str, water_coordinates: np.ndarray, water_ids: np.ndarray, ligand_pocket: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray):
    target_water_ids = _get_water_ids_inside_ligand_pocket(water_coordinates, water_ids, ligand_pocket, grid_dims, grid_origin)
    print(target_water_ids)
    filter_atoms_and_create_new_pdb(input_pdb_path=input_pdb_path, output_pdb_path=output_pdb_path, target_atom_ids=target_water_ids, type="HETATM")

def _get_water_ids_inside_ligand_pocket(water_coordinates: np.ndarray, water_ids: np.ndarray, ligand_pocket: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> Tuple[list, list]:
    _, water_coordinate_to_id = create_convert_dict(water_coordinates, water_ids, grid_origin)
    water_coordinates_inside_ligand_pocket = get_water_coordinates_inside_ligand_pocket(water_coordinates, water_ids, ligand_pocket, grid_dims, grid_origin)
    water_ids_inside_ligand_pocket = _convert_coordinates_to_water_ids(water_coordinates_inside_ligand_pocket, water_coordinate_to_id)
    return water_ids_inside_ligand_pocket

def _convert_coordinates_to_water_ids(water_coordinates: np.ndarray, water_coordinate_to_id: dict) -> list:
    target_water_ids = []
    for coordinate in water_coordinates:
        target_water_ids.append(water_coordinate_to_id[tuple(coordinate)])
    return target_water_ids


