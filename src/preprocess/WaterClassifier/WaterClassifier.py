import numpy as np
from lib.voxel import coordinate_to_voxel_index
from lib.pdb import filter_atoms_and_create_new_pdb
from lib.path import get_water_path
from modules.get_voxelized_ligand import get_voxelized_ligand
from typing import Tuple

class WaterClassifier:
    LIGAND_PRESENT_THRESHOLD = 1-np.exp(-1)

    def __init__(self, pdb_name, grid_dims, grid_origin, ligand_pocket_path):
        self.pdb_name = pdb_name
        self.grid_dims = grid_dims
        self.grid_origin = grid_origin
        self.ligand_pocket_path = ligand_pocket_path
        self.voxelized_ligand = None
        self.ligand_pocket = None
        self.water_index_to_coordinate = None
        self.water_coordinate_to_id = None

    def load_ligand(self) -> None:
        self.voxelized_ligand = get_voxelized_ligand(self.pdb_name)
    
    def define_ligand_pocket(self, ligand_pocket_definer):
        self.ligand_pocket = ligand_pocket_definer.define_ligand_pocket()
        if not np.any(self.ligand_pocket):
            raise ValueError("Ligand pocket is empty")

    def create_convert_dict(self, water_coordinates: np.ndarray) -> None:
        water_voxel_index = coordinate_to_voxel_index(water_coordinates, self.grid_origin)
        self.water_index_to_coordinate = {tuple(water_voxel_index[i]): water_coordinates[i] for i in range(water_voxel_index.shape[0])}
        self.water_coordinate_to_id = {tuple(water_coordinates[i]): i for i in range(water_coordinates.shape[0])}

    def _get_displaceable_and_non_displaceable_water_coords(self, water_classifying_rule):
        displaceable_water_coordinates, non_displaceable_water_coordinates = water_classifying_rule.classify_water()
        return displaceable_water_coordinates, non_displaceable_water_coordinates
        
    def get_classified_water_ids(self, water_coordinates: np.ndarray) -> Tuple[list, list]:
        displaceable_water_coordinates, non_displaceable_water_coordinates = self._get_displaceable_and_non_displaceable_water_coords(water_coordinates)
        displaceable_water_ids = self._convert_coordinates_to_water_ids(displaceable_water_coordinates, self.water_coordinate_to_id)
        non_displaceable_water_ids = self._convert_coordinates_to_water_ids(non_displaceable_water_coordinates, self.water_coordinate_to_id)
        return displaceable_water_ids, non_displaceable_water_ids
    
    def save_classified_water_as_pdb(self, displaceable_water_ids: list, non_displaceable_water_ids: list, output_path_displaceable: str, output_path_non_displaceable: str) -> None:
        input_pdb_path = get_water_path(self.pdb_name)
        filter_atoms_and_create_new_pdb(input_pdb_path=input_pdb_path, output_pdb_path=output_path_displaceable, target_atom_ids=displaceable_water_ids)
        filter_atoms_and_create_new_pdb(input_pdb_path=input_pdb_path, output_pdb_path=output_path_non_displaceable, target_atom_ids=non_displaceable_water_ids)
    
    def _get_voxelized_water_center(self, water_coordinates: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> np.ndarray:
        water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)

        voxelized_water_center = np.zeros(grid_dims)
        for index in water_voxel_index:
            voxelized_water_center[index[0], index[1], index[2]] = 1
        return voxelized_water_center
    
    def _is_ligand_present(self, voxelized_ligand: np.ndarray, threshold=LIGAND_PRESENT_THRESHOLD) -> np.ndarray:
        return np.any(voxelized_ligand > threshold, axis=0)
    
    def _convert_voxel_to_water_coordinates(self, water_voxelized: np.ndarray, water_index_to_coordinate: dict) -> np.ndarray:
        water_index_tmp = np.where(water_voxelized == 1)
        water_index = []
        for i in zip(water_index_tmp[0], water_index_tmp[1], water_index_tmp[2]):
            water_index.append([i[0], i[1], i[2]])

        water_coordinates = []
        for index in water_index:
            water_coordinates.append(water_index_to_coordinate[tuple(index)])
        return np.vstack(water_coordinates)


    def _convert_coordinates_to_water_ids(self, water_coordinates: np.ndarray, water_coordinate_to_id: dict) -> list:
        target_water_ids = []
        for coordinate in water_coordinates:
            target_water_ids.append(water_coordinate_to_id[tuple(coordinate)])
        return target_water_ids
