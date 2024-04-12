import numpy as np
import os
from data_loader.DataLoader import DataLoader

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
                displaceable_data1 = self._get_ndarray(pdb_name, os.path.join(training_data_dir1, 'displaceable/'))
                displaceable_data2 = self._get_ndarray(pdb_name, os.path.join(training_data_dir2, 'displaceable/'))
                # Only when the both data are not empty, add them to the list
                displaceable_data1_list.append(displaceable_data1)
                displaceable_data2_list.append(displaceable_data2)
            except (ValueError, FileNotFoundError) as e:
                print(f"Error processing {pdb_name}: {e}")
            
            try:
                non_displaceable_data1 = self._get_ndarray(pdb_name, os.path.join(training_data_dir1, 'non_displaceable/'))
                non_displaceable_data2 = self._get_ndarray(pdb_name, os.path.join(training_data_dir2, 'non_displaceable/'))
                non_displaceable_data1_list.append(non_displaceable_data1)
                non_displaceable_data2_list.append(non_displaceable_data2)
            except (ValueError, FileNotFoundError) as e:
                print(f"Error processing {pdb_name}: {e}")
        print(len(displaceable_data1_list))
        print(len(displaceable_data2_list))   
        displaceable_data1_array = np.concatenate(displaceable_data1_list, axis=0)
        displaceable_data2_array = np.concatenate(displaceable_data2_list, axis=0)
        non_displaceable_data1_array = np.concatenate(non_displaceable_data1_list, axis=0)
        non_displaceable_data2_array = np.concatenate(non_displaceable_data2_list, axis=0)
        displaceable_labels = np.ones(len(displaceable_data1_array))
        non_displaceable_labels = np.zeros(len(non_displaceable_data1_array))  

        displaceable_conbined_data = np.concatenate([displaceable_data1_array, displaceable_data2_array], axis=-1)
        non_displaceable_conbined_data = np.concatenate([non_displaceable_data1_array, non_displaceable_data2_array], axis=-1)

        all_data = np.concatenate([displaceable_conbined_data, non_displaceable_conbined_data], axis=0)
        all_labels = np.concatenate([displaceable_labels, non_displaceable_labels], axis=0)
        all_data_shuffled, all_labels_shuffled = self._shuffle_data(all_data, all_labels)

        return all_data_shuffled, all_labels_shuffled

    def load_data(self, pdb_list_path):
        with open(pdb_list_path, 'r') as f:
            pdb_list = f.read().splitlines()
            data, labels = self._get_data(pdb_list, self.training_data_dir1, self.training_data_dir2)
        return data, labels