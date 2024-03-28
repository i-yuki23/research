from WaterClassifier.WaterClassifyingRule import WaterClassifyingRule
import numpy as np
from typing import Tuple
from modules.get_voxelized_ligand import get_voxelized_ligand

class WaterClassifyingRuleCenter(WaterClassifyingRule):
    LIGAND_PRESENT_THRESHOLD = 1-np.exp(-1)

    def __init__(self, pdb_name, grid_dims, grid_origin):
        super().__init__(pdb_name, grid_dims, grid_origin)
        self.voxelized_ligand = None

    def __is_ligand_present(self, voxelized_ligand: np.ndarray, threshold=LIGAND_PRESENT_THRESHOLD) -> np.ndarray:
        return np.any(voxelized_ligand > threshold, axis=0)
    
    def __load_ligand(self) -> None:
        self.voxelized_ligand = get_voxelized_ligand(self.pdb_name)

    def classify_water(self, water_coordinates: np.ndarray, ligand_pocket: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.__load_ligand()
        voxelized_water_center = self._get_voxelized_water_center(water_coordinates, self.grid_dims, self.grid_origin)

        ligand_presence = self.__is_ligand_present(self.voxelized_ligand, self.LIGAND_PRESENT_THRESHOLD)

        displaceable_voxelized_water_center = np.where(((ligand_pocket==1) & (voxelized_water_center==1)) & ligand_presence, 1, 0)
        if not np.any(displaceable_voxelized_water_center):
            raise ValueError("No displaceable water molecules")
        self._create_convert_dict(water_coordinates)
        displaceable_water_coordinates = self._convert_voxel_to_water_coordinates(displaceable_voxelized_water_center, self.water_index_to_coordinate)

        non_displaceable_voxelized_water_center = np.where(((ligand_pocket==1) & (voxelized_water_center==1)) & (~ligand_presence), 1, 0)
        if not np.any(non_displaceable_voxelized_water_center):
            raise ValueError("No non-displaceable water molecules")
        non_displaceable_water_coordinates = self._convert_voxel_to_water_coordinates(non_displaceable_voxelized_water_center, self.water_index_to_coordinate)
        
        return displaceable_water_coordinates, non_displaceable_water_coordinates
