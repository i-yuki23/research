import numpy as np
import glob
import os
import tensorflow as tf

class TestDataLoader:

    def __init__(self, test_dir, apo_name):
        self.test_dir = test_dir
        self.apo_name = apo_name

    def get_ndarray(self):
        data_path = os.path.join(self.test_dir, f'{self.apo_name}/*.npy')
        data_path_list = glob.glob(data_path)
        if len(data_path_list) == 0:
            # print(f'No data found for {self.apo_name}')
            raise FileNotFoundError(f'No data found for {self.apo_name}')
        data_list = []
        for data_name in data_path_list:
            loaded_data = np.load(data_name)
            # print(loaded_data.shape)
            if loaded_data.size == 0:
                # print(f"This is empty data: {self.apo_name}")
                raise ValueError(f"This is empty data: {self.apo_name}")
            loaded_data_reshaped = loaded_data[np.newaxis, :, :, :, :]
            # print(loaded_data_reshaped.shape)
            data_list.append(loaded_data_reshaped)
        data_np = np.concatenate(data_list, axis=0)
        return data_np
    
    def load_data(self):
        data = self.get_ndarray()
        data_reshaped = tf.transpose(data, [0, 2, 3, 4, 1])
        return data_reshaped
    
    def get_test_data_and_water_ids(self):
        
        data_path = os.path.join(self.test_dir, f'{self.apo_name}/*.npy')
        data_path_list = glob.glob(data_path)
        if len(data_path_list) == 0:
            raise FileNotFoundError(f'No data found for {self.apo_name}')
        data_list = []
        water_ids = []
        for data_name in data_path_list:
            loaded_data = np.load(data_name)
            if loaded_data.size == 0:
                raise ValueError(f"This is empty data: {self.apo_name}")
            loaded_data_reshaped = loaded_data[np.newaxis, :, :, :, :]
            data_list.append(loaded_data_reshaped)

            file_name = os.path.basename(data_name)
            water_id = int(file_name.split('_')[2].split('.')[0])
            water_ids.append(water_id)

        data_np = np.concatenate(data_list, axis=0)
        # data_np_reshaped = tf.transpose(data_np, [0, 2, 3, 4, 1])
        return data_np, np.array(water_ids)