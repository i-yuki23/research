import numpy as np
import glob
import os
from lib.path import get_training_data_dir

# In current implementation, we save one training data for one protain, not for one water molecule. It might be better to change that.
class DataLoader:

    def __init__(self, training_data_dir):
        self.training_data_dir = training_data_dir

    def get_ndarray(self, pdb_name, training_data_path):
        data_path = os.path.join(training_data_path, f'{pdb_name}/*.npy')
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
    
    def get_test_data_and_water_ids(self, pdb_name, training_data_path):
        raise NotImplementedError("This method must be implemented in subclass")
    

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
        raise NotImplementedError("This method must be implemented in subclass")

        
