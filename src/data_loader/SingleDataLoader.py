import numpy as np
import glob
import os
from data_loader.DataLoader import DataLoader
from lib.path import get_training_data_dir

class SingleDataLoader(DataLoader):

    def __init__(self, training_data_dir):
        super().__init__(training_data_dir)

    def _get_data(self, pdb_list, training_data_dir, data_voxel_num):
        displaceable_data = self._get_ndarray(pdb_list, os.path.join(training_data_dir, 'displaceable/'), data_voxel_num)
        displaceable_labels = np.ones(len(displaceable_data))

        non_displaceable_data = self._get_ndarray(pdb_list, os.path.join(training_data_dir, 'non_displaceable/'), data_voxel_num)
        non_displaceable_labels = np.zeros(len(non_displaceable_data))

        all_data = np.concatenate([displaceable_data, non_displaceable_data], axis=0)
        all_labels = np.concatenate([displaceable_labels, non_displaceable_labels], axis=0)
        all_data_shuffled, all_labels_shuffled = self._shuffle_data(all_data, all_labels)

        return all_data_shuffled, all_labels_shuffled
