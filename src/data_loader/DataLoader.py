import numpy as np
import glob
import os
from lib.path import get_training_data_dir

class DataLoader:

    def __init__(self, training_data_dir):
        self.training_data_dir = training_data_dir

    def _get_ndarray(self, pdb_list, training_data_dir, data_voxel_num):
        data_list = []
        for pdb_name in pdb_list:
            data_path = os.path.join(training_data_dir, f'{pdb_name}/*.npy')
            data_path_list = glob.glob(data_path)
            if len(data_path_list) == 0:
                print(f'No data found for {pdb_name}')
                continue

            for data_name in data_path_list:
                loaded_data = np.load(data_name)
                loaded_data_reshaped = loaded_data.reshape((1, data_voxel_num*2+1, data_voxel_num*2+1, data_voxel_num*2+1, 1))
                data_list.append(loaded_data_reshaped)
        data_np = np.concatenate(data_list, axis=0)
        return data_np


    def _shuffle_data(self, data, labels):
        combined = list(zip(data, labels))
        np.random.shuffle(combined)
        # データとラベルを分割
        data_shuffled, labels_shuffled = zip(*combined)
        # リストをNumPy配列に変換
        data_shuffled = np.array(data_shuffled)
        labels_shuffled = np.array(labels_shuffled)
        return data_shuffled, labels_shuffled

    def _get_data(self, pdb_list, training_data_dir, data_voxel_num):
        raise NotImplementedError("This method must be implemented in subclass")


    def load_data(self, pdb_list_path, data_voxel_num):
        with open(pdb_list_path, 'r') as f:
            pdb_list = f.read().splitlines()
            data, labels = self._get_data(pdb_list, self.training_data_dir, data_voxel_num)
        return data, labels
        
