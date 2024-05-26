import numpy as np
import os
from data_loader.DataLoader import DataLoader
import tensorflow as tf

class DoubleDataLoader(DataLoader):

    def __init__(self, training_data_dir1, training_data_dir2):
        self.training_data_dir1 = training_data_dir1
        self.training_data_dir2 = training_data_dir2

    def _get_data(self, pdb_list, training_data_dir1, training_data_dir2):
        displaceable_data1_list = []
        displaceable_data2_list = []
        non_displaceable_data1_list = []
        non_displaceable_data2_list = []
        for pdb_name in pdb_list:
            try:
                displaceable_data1 = self.get_ndarray(pdb_name, os.path.join(training_data_dir1, 'displaceable/'))
                displaceable_data2 = self.get_ndarray(pdb_name, os.path.join(training_data_dir2, 'displaceable/'))
                # Only when the both data are not empty, add them to the list
                # print(displaceable_data1.shape)
                displaceable_data1_list.append(displaceable_data1)
                displaceable_data2_list.append(displaceable_data2)
            except (ValueError, FileNotFoundError) as e:
                print(f"Error processing {pdb_name}: {e}")
            
            try:
                non_displaceable_data1 = self.get_ndarray(pdb_name, os.path.join(training_data_dir1, 'non_displaceable/'))
                non_displaceable_data2 = self.get_ndarray(pdb_name, os.path.join(training_data_dir2, 'non_displaceable/'))
                non_displaceable_data1_list.append(non_displaceable_data1)
                non_displaceable_data2_list.append(non_displaceable_data2)
            except (ValueError, FileNotFoundError) as e:
                print(f"Error processing {pdb_name}: {e}")

        # Concatenate data along the data num axis
        displaceable_data1_array = np.concatenate(displaceable_data1_list, axis=0)
        displaceable_data2_array = np.concatenate(displaceable_data2_list, axis=0)
        non_displaceable_data1_array = np.concatenate(non_displaceable_data1_list, axis=0)
        non_displaceable_data2_array = np.concatenate(non_displaceable_data2_list, axis=0)
        displaceable_labels = np.ones(len(displaceable_data1_array))
        non_displaceable_labels = np.zeros(len(non_displaceable_data1_array))  

        # Concatenate data along the channel axis
        displaceable_conbined_data = np.concatenate([displaceable_data1_array, displaceable_data2_array], axis=1)
        non_displaceable_conbined_data = np.concatenate([non_displaceable_data1_array, non_displaceable_data2_array], axis=1)

        all_data = np.concatenate([displaceable_conbined_data, non_displaceable_conbined_data], axis=0)
        all_labels = np.concatenate([displaceable_labels, non_displaceable_labels], axis=0)
        all_data_shuffled, all_labels_shuffled = self._shuffle_data(all_data, all_labels)

        return all_data_shuffled, all_labels_shuffled

    def load_data(self, pdb_list_path):
        with open(pdb_list_path, 'r') as f:
            pdb_list = f.read().splitlines()
            data, labels = self._get_data(pdb_list, self.training_data_dir1, self.training_data_dir2)
            data_reshaped = tf.transpose(data, [0, 2, 3, 4, 1])
        return data_reshaped, labels
    
    def get_test_data_and_water_ids(self, pdb_name, training_data_path):
        data_path = os.path.join(training_data_path, f'{pdb_name}/*.npy')
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
        return data_np, np.array(water_ids)