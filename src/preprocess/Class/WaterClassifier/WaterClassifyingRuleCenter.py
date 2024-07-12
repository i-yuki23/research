from Class.WaterClassifier.WaterClassifyingRule import WaterClassifyingRule
import numpy as np
from typing import Tuple
from lib.helper import extract_points_within_threshold
from lib.path import get_ligand_path
from modules.get_atomic_symbol_coords_dict import get_atomic_symbol_coords_dict_from_pdb


class WaterClassifyingRuleCenter(WaterClassifyingRule):
    def __init__(self, pdb_name, grid_dims, grid_origin):
        super().__init__(pdb_name, grid_dims, grid_origin)

    
    def classify_water(self, water_coordinates: np.ndarray, ligand_pocket: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        ligand_path = get_ligand_path(self.pdb_name)
        ligand_atomic_symbol_coords_dict = get_atomic_symbol_coords_dict_from_pdb(ligand_path)

        water_coordinates_inside_ligand_pocket = self._get_water_coordinates_inside_ligand_pocket(water_coordinates, ligand_pocket)
        if water_coordinates_inside_ligand_pocket.size == 0:
            raise ValueError("No water molecules inside ligand pocket")

        temp_displaceable_water_coords = []
        for atomic_symbol, coords in ligand_atomic_symbol_coords_dict.items():
            threshold = self.RADIUSES[atomic_symbol]
            points = extract_points_within_threshold(coords, water_coordinates_inside_ligand_pocket, threshold)
            if points.size > 0:
                temp_displaceable_water_coords.append(points)

        if temp_displaceable_water_coords:
            displaceable_water_coordinates = np.vstack(temp_displaceable_water_coords)
            displaceable_water_coordinates = np.unique(displaceable_water_coordinates, axis=0)
        else:
            displaceable_water_coordinates = np.empty((0, 3))

        mask = np.ones(len(water_coordinates_inside_ligand_pocket), dtype=bool)
        for displaceable_water_coord in displaceable_water_coordinates:
            mask = mask & ~np.all(water_coordinates_inside_ligand_pocket == displaceable_water_coord, axis=1)
        non_displaceable_water_coordinates = water_coordinates_inside_ligand_pocket[mask]
        # non_displaceable_water_coordinates = np.array(list(set(map(tuple, water_coordinates_inside_ligand_pocket)) - set(map(tuple, displaceable_water_coordinates))))

        return displaceable_water_coordinates, non_displaceable_water_coordinates
