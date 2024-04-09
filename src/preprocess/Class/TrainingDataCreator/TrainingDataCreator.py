import numpy as np
from modules.get_labeled_water_coords import get_displaceable_water_coords, get_non_displaceable_water_coords
from lib.path import get_training_data_path
from lib.helper import make_dir
from lib.voxel import get_voxel_info
from typing import Tuple

class TrainingDataCreator:

    def __init__(self, pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer, data_voxel_num):
        self.pdb_name = pdb_name
        self.ligand_voxel_num = ligand_voxel_num
        self.classifying_rule = classifying_rule
        self.ligand_pocket_definer = ligand_pocket_definer
        self.data_voxel_num = data_voxel_num
        self.grid_origin = None
        self.grid_dims = None
        self.displaceable_save_path = None
        self.non_displaceable_save_path = None
        self._set_grid_info()
        self._set_save_path()

    def _set_grid_info(self):
        self.grid_dims, self.grid_origin = get_voxel_info(self.pdb_name)

    def _set_save_path(self):
        raise NotImplementedError("This method must be implemented in subclass")

    def _get_taregt_water_molecules(self):

        displaceable_water_coords = get_displaceable_water_coords(self.pdb_name, self.ligand_voxel_num, self.classifying_rule, self.ligand_pocket_definer)
        non_displaceable_water_coords = get_non_displaceable_water_coords(self.pdb_name, self.ligand_voxel_num, self.classifying_rule, self.ligand_pocket_definer)

        return displaceable_water_coords, non_displaceable_water_coords
    
    def _get_base_voxel_data(self):
        raise NotImplementedError("This method must be implemented in subclass")
    
    def _get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("This method must be implemented in subclass")

    def save_training_data(self) -> None:

        training_data_displaceable, training_data_non_displaceable = self._get_training_data()

        make_dir(self.displaceable_save_path)
        np.save(self.displaceable_save_path, training_data_displaceable)
    
        make_dir(self.non_displaceable_save_path)
        np.save(self.non_displaceable_save_path, training_data_non_displaceable)





