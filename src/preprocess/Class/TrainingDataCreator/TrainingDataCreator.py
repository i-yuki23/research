# 

import numpy as np
from modules.get_labeled_water import get_displaceable_water_coords, get_non_displaceable_water_coords, get_displaceable_water_ids, get_non_displaceable_water_ids
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
        self._set_grid_info()

    def _set_grid_info(self):
        self.grid_dims, self.grid_origin = get_voxel_info(self.pdb_name)

    def _get_save_path_dis(self):
        raise NotImplementedError("This method must be implemented in subclass")
    
    def _get_save_path_non_dis(self):
        raise NotImplementedError("This method must be implemented in subclass")

    def _get_taregt_water_coords(self):

        displaceable_water_coords = get_displaceable_water_coords(self.pdb_name, self.ligand_voxel_num, self.classifying_rule, self.ligand_pocket_definer)
        non_displaceable_water_coords = get_non_displaceable_water_coords(self.pdb_name, self.ligand_voxel_num, self.classifying_rule, self.ligand_pocket_definer)
        return displaceable_water_coords, non_displaceable_water_coords
    
    def _get_taregt_water_ids(self):
        displaceable_water_ids = get_displaceable_water_ids(self.pdb_name, self.ligand_voxel_num, self.classifying_rule, self.ligand_pocket_definer)
        non_displaceable_water_ids = get_non_displaceable_water_ids(self.pdb_name, self.ligand_voxel_num, self.classifying_rule, self.ligand_pocket_definer)
        return displaceable_water_ids, non_displaceable_water_ids

    def _get_base_voxel_data(self):
        raise NotImplementedError("This method must be implemented in subclass")
    
    def _get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("This method must be implemented in subclass")

    # save training for each water molecule and name it with its water_id
    def save_training_data(self) -> None:

        training_data_displaceable, training_data_non_displaceable = self._get_training_data()
        displaceable_water_ids, non_displaceable_water_ids = self._get_taregt_water_ids()

        for index, one_training_data_displaceable in enumerate(training_data_displaceable):
            save_path_dis = self._get_save_path_dis(displaceable_water_ids[index])
            make_dir(save_path_dis)
            np.save(save_path_dis, one_training_data_displaceable[:, :, :, :])
    
        for index, one_training_data_non_displaceable in enumerate(training_data_non_displaceable):
            save_path_non_dis = self._get_save_path_non_dis(non_displaceable_water_ids[index])
            make_dir(save_path_non_dis)
            np.save(save_path_non_dis, one_training_data_non_displaceable[:, :, :, :])





