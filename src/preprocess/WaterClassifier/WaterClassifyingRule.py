import numpy as np
from typing import Tuple
from lib.voxel import coordinate_to_voxel_index
from lib.helper import extract_points_within_threshold
from modules.get_atomic_symbol_coords_dict import get_atomic_symbol_coords_dict_from_pdb
from lib.path import get_ligand_path

class WaterClassifyingRule:
    def __init__(self, pdb_name, grid_dims, grid_origin):
        self.pdb_name = pdb_name
        self.grid_dims = grid_dims
        self.grid_origin = grid_origin
        self.water_index_to_coordinate = None
        self.water_coordinate_to_id = None

    def _create_convert_dict(self, water_coordinates: np.ndarray) -> None:
        water_voxel_index = coordinate_to_voxel_index(water_coordinates, self.grid_origin)
        self.water_index_to_coordinate = {tuple(water_voxel_index[i]): water_coordinates[i] for i in range(water_voxel_index.shape[0])}
        self.water_coordinate_to_id = {tuple(water_coordinates[i]): i for i in range(water_coordinates.shape[0])}

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
    
    def classify_water(self, water_coordinates: np.ndarray, ligand_pocket: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("This method must be implemented in subclass")