import sys
sys.path.append('../..')

from lib.path import get_test_gr_path
from lib.voxel import get_voxel_info
import numpy as np
from lib.dx import read_dx
from lib.helper import make_dir
from modules.fetch_neighboring_voxel import fetch_neighboring_voxel

class VoxelDataExtractor:

    def __init__(self, protein_name, apo_name, holo_name, base_voxel_path, data_voxel_num):
        self.protein_name = protein_name
        self.apo_name = apo_name
        self.holo_name = holo_name
        self.data_voxel_num = data_voxel_num
        self.grid_dims, self.grid_origin = get_voxel_info(dx_path=get_test_gr_path(protein_name, apo_name))
        self.base_voxel = self.__get_base_voxel_data(base_voxel_path)

    def __get_base_voxel_data(self, base_voxel_path):
        base_voxel, _, _ = read_dx(base_voxel_path)
        return base_voxel

    def __extract_test_voxel_data(self, water_coords):
        test_voxel_data_list = []
        for water_coordinate in water_coords:
            test_voxel_data = fetch_neighboring_voxel(water_coordinate, self.base_voxel, self.grid_origin, self.grid_dims, self.data_voxel_num)
            test_voxel_data_list.append(test_voxel_data[:, :, :, np.newaxis])
        test_voxel_data_array = np.array(test_voxel_data_list)
        return test_voxel_data_array

    def save_test_data(self, water_coords, water_ids) -> None:
        test_data = self.__extract_test_voxel_data(water_coords)

        for index, one_test_data in enumerate(test_data):
            save_path = f"/mnt/ito/test/{self.protein_name}/test_data/{self.apo_name}/{self.holo_name}/water_id_{water_ids[index]}.npy"
            make_dir(save_path)
            np.save(save_path, one_test_data[:, :, :, :])

