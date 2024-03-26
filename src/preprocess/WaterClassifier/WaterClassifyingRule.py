import numpy as np
from typing import Tuple
from lib.voxel import coordinate_to_voxel_index
from lib.helper import extract_points_within_threshold
from modules.get_atomic_symbol_coords_dict import get_atomic_symbol_coords_dict_from_pdb
from lib.path import get_ligand_path

class WaterClassifyingRule:
    def __init__(self, rule, pdb_name):
        self.rule = rule
        self.pdb_name = pdb_name

    def _get_voxelized_water_center(self, water_coordinates: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> np.ndarray:
        water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)

        voxelized_water_center = np.zeros(grid_dims)
        for index in water_voxel_index:
            voxelized_water_center[index[0], index[1], index[2]] = 1
        return voxelized_water_center
    
    def _convert_voxel_to_water_coordinates(self, water_voxelized: np.ndarray, water_index_to_coordinate: dict) -> np.ndarray:
        water_index_tmp = np.where(water_voxelized == 1)
        water_index = []
        for i in zip(water_index_tmp[0], water_index_tmp[1], water_index_tmp[2]):
            water_index.append([i[0], i[1], i[2]])

        water_coordinates = []
        for index in water_index:
            water_coordinates.append(water_index_to_coordinate[tuple(index)])
        return np.vstack(water_coordinates)