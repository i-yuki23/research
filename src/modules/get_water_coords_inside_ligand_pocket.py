import sys 
sys.path.append('..')

from typing import Tuple
import numpy as np
from lib.voxel import coordinate_to_voxel_index, get_voxel_info
from lib.pdb import get_coordinates_from_pdb, filter_atoms_and_create_new_pdb

def create_convert_dict(water_coordinates: np.ndarray, grid_origin: np.ndarray) -> None:
    water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)
    water_index_to_coordinate = {tuple(water_voxel_index[i]): water_coordinates[i] for i in range(water_voxel_index.shape[0])}
    water_coordinate_to_id = {tuple(water_coordinates[i]): i for i in range(water_coordinates.shape[0])}
    return water_index_to_coordinate, water_coordinate_to_id

def get_voxelized_water_center(water_coordinates: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> np.ndarray:
    water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)

    voxelized_water_center = np.zeros(grid_dims)
    for index in water_voxel_index:
        voxelized_water_center[index[0], index[1], index[2]] = 1
    return voxelized_water_center
    

def convert_voxel_to_water_coordinates(water_voxelized: np.ndarray, water_index_to_coordinate: dict) -> np.ndarray:
    water_index_tmp = np.where(water_voxelized == 1)
    water_index = []
    for i in zip(water_index_tmp[0], water_index_tmp[1], water_index_tmp[2]):
        water_index.append([i[0], i[1], i[2]])

    water_coordinates = []
    for index in water_index:
        water_coordinates.append(water_index_to_coordinate[tuple(index)])
    return np.vstack(water_coordinates)

def get_water_coordinates_inside_ligand_pocket(water_coordinates: np.ndarray, ligand_pocket: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> np.ndarray:
    voxelized_water_center = get_voxelized_water_center(water_coordinates, grid_dims, grid_origin)
    voxelized_water_center_inside_ligand_pocket = np.where((voxelized_water_center == 1) & (ligand_pocket == 1), 1, 0)
    water_index_to_coordinate, _ = create_convert_dict(water_coordinates, grid_origin)
    water_coordinates_inside_ligand_pocket = convert_voxel_to_water_coordinates(voxelized_water_center_inside_ligand_pocket, water_index_to_coordinate)

    return water_coordinates_inside_ligand_pocket



def save_water_coordinates_inside_ligand_pocket(input_pdb_path: str, output_pdb_path: str, water_coordinates: np.ndarray, ligand_pocket: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray):
    water_ids = _get_water_ids_inside_ligand_pocket(water_coordinates, ligand_pocket, grid_dims, grid_origin)
    water_ids = [i + 2344 for i in water_ids]
    print(water_ids)
    filter_atoms_and_create_new_pdb(input_pdb_path=input_pdb_path, output_pdb_path=output_pdb_path, target_atom_ids=water_ids, type="HETATM")

def _get_water_ids_inside_ligand_pocket(water_coordinates: np.ndarray, ligand_pocket: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> Tuple[list, list]:
    _, water_coordinate_to_id = create_convert_dict(water_coordinates, grid_origin)
    water_coordinates_inside_ligand_pocket = get_water_coordinates_inside_ligand_pocket(water_coordinates, ligand_pocket, grid_dims, grid_origin)
    water_ids_inside_ligand_pocket = _convert_coordinates_to_water_ids(water_coordinates_inside_ligand_pocket, water_coordinate_to_id)
    return water_ids_inside_ligand_pocket

def _convert_coordinates_to_water_ids(water_coordinates: np.ndarray, water_coordinate_to_id: dict) -> list:
    target_water_ids = []
    for coordinate in water_coordinates:
        target_water_ids.append(water_coordinate_to_id[tuple(coordinate)])
    return target_water_ids

test_dir = "/mnt/ito/test/WD5/"
pdb_name = "2h14"
holo_name = "3smrA"

dx_path = test_dir + pdb_name + "_min.dx"
ligand_pocket_path = test_dir + "ligand_pocket/" + holo_name + ".npy"
water_path = test_dir + f"{pdb_name}_apo_HOH.pdb"

water_coordinates = get_coordinates_from_pdb(water_path, type="HETATM")
print(water_coordinates.shape)
ligand_pocket = np.load(ligand_pocket_path)
grid_dims, grid_origin = get_voxel_info(dx_path=dx_path)


input_pdb_path = water_path
output_pdb_path = test_dir + f"water_inside_ligand_pocket/{pdb_name}_apo/" + "HOH.pdb"
print(output_pdb_path)

save_water_coordinates_inside_ligand_pocket(input_pdb_path, output_pdb_path, water_coordinates, ligand_pocket, grid_dims, grid_origin)