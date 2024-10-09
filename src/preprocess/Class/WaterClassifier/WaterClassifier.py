import numpy as np
from lib.voxel import coordinate_to_voxel_index
from lib.pdb import filter_atoms_and_create_new_pdb
from lib.path import get_water_path
from typing import Tuple

class WaterClassifier:
    LIGAND_PRESENT_THRESHOLD = 1-np.exp(-1)

    def __init__(self, pdb_name, grid_dims, grid_origin, ligand_path):
        self.pdb_name = pdb_name
        self.grid_dims = grid_dims
        self.grid_origin = grid_origin
        self.ligand_path = ligand_path
        self.ligand_pocket = None
        self.water_index_to_coordinate = None
        self.water_coordinate_to_id = None
    
    def define_ligand_pocket(self, ligand_pocket_definer):
        self.ligand_pocket = ligand_pocket_definer.define_ligand_pocket()
        if not np.any(self.ligand_pocket):
            raise ValueError("Ligand pocket is empty")

    def create_convert_dict(self, water_coordinates: np.ndarray) -> None:
        water_voxel_index = coordinate_to_voxel_index(water_coordinates, self.grid_origin)
        self.water_index_to_coordinate = {tuple(water_voxel_index[i]): water_coordinates[i] for i in range(water_voxel_index.shape[0])}
        self.water_coordinate_to_id = {tuple(water_coordinates[i]): i for i in range(water_coordinates.shape[0])}

    def _get_displaceable_and_non_displaceable_water_coords(self, water_coordinates, water_classifying_rule):
        displaceable_water_coordinates, non_displaceable_water_coordinates = water_classifying_rule.classify_water(self.ligand_path, water_coordinates, self.ligand_pocket)
        return displaceable_water_coordinates, non_displaceable_water_coordinates
        
    def get_classified_water_ids(self, water_coordinates: np.ndarray, water_classifying_rule) -> Tuple[list, list]:
        displaceable_water_coordinates, non_displaceable_water_coordinates = self._get_displaceable_and_non_displaceable_water_coords(water_coordinates, water_classifying_rule)
        displaceable_water_ids = self._convert_coordinates_to_water_ids(displaceable_water_coordinates, self.water_coordinate_to_id)
        non_displaceable_water_ids = self._convert_coordinates_to_water_ids(non_displaceable_water_coordinates, self.water_coordinate_to_id)
        return displaceable_water_ids, non_displaceable_water_ids
    
    def save_classified_water_as_pdb(self, input_pdb_path: str, displaceable_water_ids: list, non_displaceable_water_ids: list, output_path_displaceable: str, output_path_non_displaceable: str) -> None:
        filter_atoms_and_create_new_pdb(input_pdb_path=input_pdb_path, output_pdb_path=output_path_displaceable, target_atom_ids=displaceable_water_ids)
        filter_atoms_and_create_new_pdb(input_pdb_path=input_pdb_path, output_pdb_path=output_path_non_displaceable, target_atom_ids=non_displaceable_water_ids)
    
    def _convert_coordinates_to_water_ids(self, water_coordinates: np.ndarray, water_coordinate_to_id: dict) -> list:
        target_water_ids = []
        for coordinate in water_coordinates:
            target_water_ids.append(water_coordinate_to_id[tuple(coordinate)])
        return target_water_ids
