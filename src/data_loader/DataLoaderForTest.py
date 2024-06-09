import numpy as np
import glob
import os
from data_loader.DataLoader import DataLoader
import tensorflow as tf

class DataLoaderForTest(DataLoader):

    def __init__(self, test_dir):
        self.test_dir = test_dir

    def get_ndarray(self, pdb_name):
        data_path = os.path.join(self.test_dir, f'{pdb_name}/*.npy')
        data_path_list = glob.glob(data_path)
        if len(data_path_list) == 0:
            # print(f'No data found for {pdb_name}')
            raise FileNotFoundError(f'No data found for {pdb_name}')
        data_list = []
        for data_name in data_path_list:
            loaded_data = np.load(data_name)
            # print(loaded_data.shape)
            if loaded_data.size == 0:
                # print(f"This is empty data: {pdb_name}")
                raise ValueError(f"This is empty data: {pdb_name}")
            loaded_data_reshaped = loaded_data[np.newaxis, :, :, :, :]
            # print(loaded_data_reshaped.shape)
            data_list.append(loaded_data_reshaped)
        data_np = np.concatenate(data_list, axis=0)
        return data_np
    
    def load_data(self, pdb_name):
        data = self.get_ndarray(pdb_name)
        data_reshaped = tf.transpose(data, [0, 2, 3, 4, 1])
        return data_reshaped
    
    def get_test_data_and_water_ids(self, pdb_name):
        
        data_path = os.path.join(self.test_dir, f'{pdb_name}/*.npy')
        data_path_list = glob.glob(data_path)
        if len(data_path_list) == 0:
            raise FileNotFoundError(f'No data found for {pdb_name}')
        data_list = []
        water_ids = []
        for data_name in data_path_list:
            loaded_data = np.load(data_name)
            if loaded_data.size == 0:
                raise ValueError(f"This is empty data: {pdb_name}")
            loaded_data_reshaped = loaded_data[np.newaxis, :, :, :, :]
            data_list.append(loaded_data_reshaped)

            file_name = os.path.basename(data_name)
            water_id = int(file_name.split('_')[2].split('.')[0])
            water_ids.append(water_id)

        data_np = np.concatenate(data_list, axis=0)
        # data_np_reshaped = tf.transpose(data_np, [0, 2, 3, 4, 1])
        return data_np, np.array(water_ids)