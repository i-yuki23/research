from Class.WaterClassifier.WaterClassifyingRule import WaterClassifyingRule
import numpy as np
from typing import Tuple
from lib.helper import extract_points_within_threshold
from modules.get_atomic_symbol_coords_dict import get_atomic_symbol_coords_dict_from_pdb
from lib.path import get_ligand_path

class WaterClassifyingRuleEmbedding(WaterClassifyingRule):
    EMBEDDING_DIST = {'C': -0.18937803424481814, 'N': 0.3135110597898958, 'O': 0.25102997293643625, 'S': -0.6729120085686606, 'B' : 0, 'F' : -0.0854913016297294, 'P' : -0.39769009362721164, 'I' : 0, 'Cl' : -0.3519097967906015,'Br': 0}
    def __init__(self, pdb_name, grid_dims, grid_origin):
        super().__init__(pdb_name, grid_dims, grid_origin)
    
    def classify_water(self, ligand_path: str, water_coordinates: np.ndarray, ligand_pocket: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        ligand_atomic_symbol_coords_dict = get_atomic_symbol_coords_dict_from_pdb(ligand_path)

        water_coordinates_inside_ligand_pocket = self._get_water_coordinates_inside_ligand_pocket(water_coordinates, ligand_pocket)
        if water_coordinates_inside_ligand_pocket.size == 0:
            raise ValueError("No water molecules inside ligand pocket")

        temp_displaceable_water_coords = []
        for atomic_symbol, coords in ligand_atomic_symbol_coords_dict.items():
            # subtract embedding distance from threshold
            threshold = self.RADIUSES[atomic_symbol] + self.RADIUSES['O'] - self.EMBEDDING_DIST[atomic_symbol]
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
