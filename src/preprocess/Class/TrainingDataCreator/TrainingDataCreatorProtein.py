from Class.TrainingDataCreator.TrainingDataCreator import TrainingDataCreator
import numpy as np
from modules.fetch_neighboring_voxel import fetch_neighboring_voxel
from modules.get_voxelized_protein import get_voxelized_protein
from lib.path import get_gr_path, get_training_data_path
from lib.dx import read_dx

class TrainingDataCreatorProtein(TrainingDataCreator):
    
    def __init__(self, pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer, data_voxel_num):
        super().__init__(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer, data_voxel_num)

    def _get_save_path_dis(self, water_id):
        return get_training_data_path('Protein', 'displaceable', self.data_voxel_num, self.classifying_rule, self.ligand_pocket_definer, self.ligand_voxel_num, self.pdb_name, water_id)

    def _get_save_path_non_dis(self, water_id):
        return get_training_data_path('Protein', 'non_displaceable', self.data_voxel_num, self.classifying_rule, self.ligand_pocket_definer, self.ligand_voxel_num, self.pdb_name, water_id)

    def _get_base_voxel_data(self):
        return get_voxelized_protein(self.pdb_name)
    
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
        potein_voxel = self._get_base_voxel_data()
        training_data_displaceable = self.__extract_training_voxel_data(displaceable_water_coords, potein_voxel)
        training_data_non_displaceable = self.__extract_training_voxel_data(non_displaceable_water_coords, potein_voxel)
        return training_data_displaceable, training_data_non_displaceable
    