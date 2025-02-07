import numpy as np
from lib.pdb import get_coordinates_from_pdb, get_atom_id_from_pdb
from lib.helper import make_dir
from typing import Tuple

class TrainingDataCreator:

    def __init__(self, data_voxel_num, grid_origin, grid_dims, save_dir_dis, save_dir_non_dis, base_voxel_data_path, displaceable_water_path, non_displaceable_water_path):
        self.data_voxel_num = data_voxel_num
        self.save_dir_dis = save_dir_dis
        self.save_dir_non_dis = save_dir_non_dis
        self.base_voxel_data_path = base_voxel_data_path
        self.displaceable_water_path = displaceable_water_path
        self.non_displaceable_water_path = non_displaceable_water_path
        self.grid_origin = grid_origin
        self.grid_dims = grid_dims

    def _get_save_path_dis(self):
        raise NotImplementedError("This method must be implemented in subclass")
    
    def _get_save_path_non_dis(self):
        raise NotImplementedError("This method must be implemented in subclass")

    def _get_taregt_water_coords(self):

        displaceable_water_coords = get_coordinates_from_pdb(self.displaceable_water_path)
        non_displaceable_water_coords = get_coordinates_from_pdb(self.non_displaceable_water_path)
        return displaceable_water_coords, non_displaceable_water_coords
    
    def _get_taregt_water_ids(self):
        displaceable_water_ids = get_atom_id_from_pdb(self.displaceable_water_path, 'ATOM')
        non_displaceable_water_ids = get_atom_id_from_pdb(self.non_displaceable_water_path, 'ATOM')
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





