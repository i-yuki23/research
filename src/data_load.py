import numpy as np
import glob
import os
from lib.path import get_training_data_dir
import warnings


def get_ndarray(pdb_list, base_path, data_voxel_num):
    data_list = []
    for pdb_name in pdb_list:
        data_path = os.path.join(base_path, f'{pdb_name}/*.npy')
        data_path_list = glob.glob(data_path)
        if len(data_path_list) == 0:
            # print(f'Water is not found in {base_path}')
            continue  

        for data_name in data_path_list:
            loaded_data = np.load(data_name)
            loaded_data_reshaped = loaded_data.reshape((1, data_voxel_num*2+1, data_voxel_num*2+1, data_voxel_num*2+1, 1))
            data_list.append(loaded_data_reshaped)
    data_np = np.concatenate(data_list, axis=0)
    return data_np


def shuffle_data(data, labels):
    combined = list(zip(data, labels))
    np.random.shuffle(combined)
    # データとラベルを分割
    data_shuffled, labels_shuffled = zip(*combined)
    # リストをNumPy配列に変換
    data_shuffled = np.array(data_shuffled)
    labels_shuffled = np.array(labels_shuffled)
    return data_shuffled, labels_shuffled

def get_data(pdb_list, base_path, data_voxel_num):
    displaceable_data = get_ndarray(pdb_list, os.path.join(base_path, 'displaceable/'), data_voxel_num)
    displaceable_labels = np.ones(len(displaceable_data))

    non_displaceable_data = get_ndarray(pdb_list, os.path.join(base_path, 'non_displaceable/'), data_voxel_num)
    non_displaceable_labels = np.zeros(len(non_displaceable_data))

    all_data = np.concatenate([displaceable_data, non_displaceable_data], axis=0)
    all_labels = np.concatenate([displaceable_labels, non_displaceable_labels], axis=0)
    all_data_shuffled, all_labels_shuffled = shuffle_data(all_data, all_labels)

    return all_data_shuffled, all_labels_shuffled


def load_data(pdb_list_path, training_data_dir, data_voxel_num):
    with open(pdb_list_path, 'r') as f:
        pdb_list = f.read().splitlines()
        data, labels = get_data(pdb_list, training_data_dir, data_voxel_num)
    return data, labels
        
