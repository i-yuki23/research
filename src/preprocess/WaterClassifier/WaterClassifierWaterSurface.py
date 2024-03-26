from WaterClassifier.WaterClassifier import WaterClassifier
import numpy as np
from typing import Tuple
from lib.helper import extract_points_within_threshold
from modules.get_atomic_symbol_coords_dict import get_atomic_symbol_coords_dict_from_pdb
from lib.path import get_ligand_path


class WaterClassifierWaterSurface(WaterClassifier):
    RADIUSES = {'C': 1.69984, 'N': 1.62500, 'O': 1.51369, 'S': 1.78180, 'H': 1.2, 'B' : 1.92, 'F' : 1.47, 'P' : 1.80, 'I' : 1.98}

    def __init__(self, pdb_name, grid_dims, grid_origin, ligand_pocket_path):
        super().__init__(pdb_name, grid_dims, grid_origin, ligand_pocket_path)

    def __get_water_coordinates_inside_ligand_pocket(self, water_coordinates: np.ndarray) -> np.ndarray:

        voxelized_water_center = self._get_voxelized_water_center(water_coordinates, self.grid_dims, self.grid_origin)
        voxelized_water_center_inside_ligand_pocket = np.where((voxelized_water_center == 1) & (self.ligand_pocket == 1), 1, 0)
        water_coordinates_inside_ligand_pocket = self._convert_voxel_to_water_coordinates(voxelized_water_center_inside_ligand_pocket, self.water_index_to_coordinate)

        return water_coordinates_inside_ligand_pocket
    
    def _get_displaceable_and_non_displaceable_water_coords(self, water_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        ligand_path = get_ligand_path(self.pdb_name)
        ligand_atomic_symbol_coords_dict = get_atomic_symbol_coords_dict_from_pdb(ligand_path)

        water_coordinates_inside_ligand_pocket = self.__get_water_coordinates_inside_ligand_pocket(water_coordinates)
        if water_coordinates_inside_ligand_pocket.size == 0:
            raise ValueError("No water molecules inside ligand pocket")

        displaceable_water_coordinates = np.array([]).reshape(0, 3)  
        for atomic_symbol, coords in ligand_atomic_symbol_coords_dict.items():
            threshold = self.RADIUSES[atomic_symbol] + self.RADIUSES['O']
            points = extract_points_within_threshold(coords, water_coordinates_inside_ligand_pocket, threshold)
            if points.size > 0:
                displaceable_water_coordinates = np.vstack((displaceable_water_coordinates, points))
        displaceable_water_coordinates = np.unique(displaceable_water_coordinates, axis=0)
        non_displaceable_water_coordinates = np.array(list(set(map(tuple, water_coordinates_inside_ligand_pocket)) - set(map(tuple, displaceable_water_coordinates))))

        return displaceable_water_coordinates, non_displaceable_water_coordinates