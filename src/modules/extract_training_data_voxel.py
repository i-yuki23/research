import numpy as np
from lib.voxel import coordinate_to_voxel_index

def extract_training_data_voxel(water_coordinate, gr_or_gist, grid_origin, grid_dims, data_voxel_num):
    water_voxel_index = coordinate_to_voxel_index(water_coordinate, grid_origin)
    # 各次元におけるスライス範囲を計算
    slices = []
    for i in range(3):
        start = max(0, water_voxel_index[i] - data_voxel_num)
        end = min(grid_dims[i], water_voxel_index[i] + data_voxel_num + 1)
        slices.append(slice(start, end))
    
    gr_or_gist_data = gr_or_gist[slices[0], slices[1], slices[2]]

    # 必要なパディングを計算
    padding = [(max(0, data_voxel_num - water_voxel_index[i]),
                max(0, (water_voxel_index[i] + data_voxel_num + 1) - grid_dims[i]))
                for i in range(3)]
    # print("パディング:", padding)
    # パディングを適用
    gr_or_gist_data_padded = np.pad(gr_or_gist_data, padding, mode='constant', constant_values=0)
    
    return gr_or_gist_data_padded