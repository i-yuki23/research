from Class.TrainingDataCreator.TrainingDataCreator import TrainingDataCreator
import numpy as np
import os
from modules.fetch_neighboring_voxel import fetch_neighboring_voxel
from modules.get_voxelized_protein import get_voxelized_protein

class TrainingDataCreatorProtein(TrainingDataCreator):
    
    def __init__(self, data_voxel_num, grid_origin, grid_dims, save_dir_dis, save_dir_non_dis, base_voxel_data_path, displaceable_water_path, non_displaceable_water_path):
        super().__init__(data_voxel_num, grid_origin, grid_dims, save_dir_dis, save_dir_non_dis, base_voxel_data_path, displaceable_water_path, non_displaceable_water_path)

    def _get_save_path_dis(self, water_id):
        return os.path.join(self.save_dir_dis, f'water_id_{water_id}.npy')

    def _get_save_path_non_dis(self, water_id):
        return os.path.join(self.save_dir_non_dis, f'water_id_{water_id}.npy')

    def _get_base_voxel_data(self):
        return get_voxelized_protein(self.base_voxel_data_path, self.grid_origin, self.grid_dims)
    
    def __extract_training_voxel_data(self, water_coordinates, base_voxel):
        training_voxel_data_list = []
        for water_coordinate in water_coordinates:
            training_voxel_data = []
            for one_channel in base_voxel:
                training_voxel_data.append(fetch_neighboring_voxel(water_coordinate, one_channel, self.grid_origin, self.grid_dims, self.data_voxel_num))
            training_voxel_data = np.array(training_voxel_data)
            training_voxel_data_list.append(training_voxel_data)
        training_voxel_data_array = np.array(training_voxel_data_list)
        return training_voxel_data_array
    
    def _get_training_data(self):

        displaceable_water_coords, non_displaceable_water_coords = self._get_taregt_water_coords()
        protein_voxel = self._get_base_voxel_data()
        training_data_displaceable = self.__extract_training_voxel_data(displaceable_water_coords, protein_voxel)
        training_data_non_displaceable = self.__extract_training_voxel_data(non_displaceable_water_coords, protein_voxel)
        return training_data_displaceable, training_data_non_displaceable