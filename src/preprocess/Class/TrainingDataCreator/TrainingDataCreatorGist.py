from Class.TrainingDataCreator.TrainingDataCreator import TrainingDataCreator
import numpy as np
from lib.path import get_gist_path, get_training_data_path
from lib.dx import read_dx
from modules.voxelizer import voxelizer_atom
from modules.fetch_neighboring_voxel import fetch_neighboring_voxel


class TrainingDataCreatorGist(TrainingDataCreator):
    WATER_CHANNEL_INDEX = 2
    WATER_PRESENCE_THRESHOLD = 10**(-6)
    
    def __init__(self, pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer, data_voxel_num):
        super().__init__(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer, data_voxel_num)

    def _set_save_path(self):
        self.displaceable_save_path = get_training_data_path('gist', 'displaceable', self.data_voxel_num, self.classifying_rule, self.ligand_pocket_definer, self.ligand_voxel_num, self.pdb_name)
        self.non_displaceable_save_path = get_training_data_path('gist', 'non_displaceable', self.data_voxel_num, self.classifying_rule, self.ligand_pocket_definer, self.ligand_voxel_num, self.pdb_name)

    def _get_base_voxel_data(self):
        gist_map_path = get_gist_path(self.pdb_name)
        gist_map, _, _ = read_dx(gist_map_path)
        return gist_map

        
    def __get_base_gist_map_for_one_water(self, water_coord: np.ndarray, gist_map: np.ndarray) -> np.ndarray:
        one_water_voxelized = voxelizer_atom(
            atomic_symbols=['O'],
            atom_coordinates=[water_coord],  # atom_coordinatesは2次元配列を受け取るから[]が必要
            grid_origin=self.grid_origin,
            grid_dims=self.grid_dims,
        )
        base_gist_map_in_water = np.where((one_water_voxelized[self.WATER_CHANNEL_INDEX] > self.WATER_PRESENCE_THRESHOLD), gist_map, 0)

        return base_gist_map_in_water
    

    def _get_training_data(self):

        gist_map = self._get_base_voxel_data()
        displaceable_water_coords, non_displaceable_water_coords = self._get_taregt_water_molecules()

        training_data_displaceable_list = []
        training_data_non_displaceable_list = []
        for displaceable_water_coord, non_displaceable_water_coord in zip(displaceable_water_coords, non_displaceable_water_coords):
            displaceable_base_gist_map = self.__get_base_gist_map_for_one_water(displaceable_water_coord, gist_map)
            non_displaceable_base_gist_map = self.__get_base_gist_map_for_one_water(non_displaceable_water_coord, gist_map)

            displaceable_training_voxel_data = fetch_neighboring_voxel(displaceable_water_coord, displaceable_base_gist_map, self.grid_origin, self.grid_dims, self.data_voxel_num)
            non_displaceable_training_voxel_data = fetch_neighboring_voxel(non_displaceable_water_coord, non_displaceable_base_gist_map, self.grid_origin, self.grid_dims, self.data_voxel_num)

            training_data_displaceable_list.append(displaceable_training_voxel_data)
            training_data_non_displaceable_list.append(non_displaceable_training_voxel_data)

        training_data_displaceable = np.array(training_data_displaceable_list)
        training_data_non_displaceable = np.array(training_data_non_displaceable_list)

        return training_data_displaceable, training_data_non_displaceable
