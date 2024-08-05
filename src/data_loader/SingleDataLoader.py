import numpy as np
import glob
import os
from data_loader.DataLoader import DataLoader
from lib.path import get_training_data_dir
import tensorflow as tf

class SingleDataLoader(DataLoader):

    def __init__(self, training_data_dir):
        super().__init__(training_data_dir)

    def __get_all_ndarray(self, pdb_list, training_data_path):
        all_data = []
        for pdb_name in pdb_list:
            try:
                data = self.get_ndarray(pdb_name, training_data_path)
                all_data.append(data)
            except Exception as e:
                print(f"Error processing {pdb_name}: {e}")
        all_data_array = np.concatenate(all_data, axis=0)
        return all_data_array

    def _get_data(self, pdb_list):
        displaceable_data = self.__get_all_ndarray(pdb_list, os.path.join(self.training_data_dir, 'displaceable/'))
        displaceable_labels = np.ones(len(displaceable_data))

        non_displaceable_data = self.__get_all_ndarray(pdb_list, os.path.join(self.training_data_dir, 'non_displaceable/'))
        non_displaceable_labels = np.zeros(len(non_displaceable_data))

        all_data = np.concatenate([displaceable_data, non_displaceable_data], axis=0)
        all_labels = np.concatenate([displaceable_labels, non_displaceable_labels], axis=0)
        all_data_shuffled, all_labels_shuffled = self._shuffle_data(all_data, all_labels)

        return all_data_shuffled, all_labels_shuffled

    def load_data(self, pdb_list_path):
        with open(pdb_list_path, 'r') as f:
            pdb_list = f.read().splitlines()
            data, labels = self._get_data(pdb_list)
            # data_reshaped = tf.transpose(data, [0, 2, 3, 4, 1])
        return data, labels
    
    def load_data_with_pdb_array(self, pdb_array):
        data, labels = self._get_data(pdb_array)
        # data_reshaped = tf.transpose(data, [0, 2, 3, 4, 1])
        return data, labels
    
    def get_test_data_and_water_ids(self, pdb_name, dis_or_non):
        test_data_dir = os.path.join(self.training_data_dir, dis_or_non)
        
        data_path = os.path.join(test_data_dir, f'{pdb_name}/*.npy')
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