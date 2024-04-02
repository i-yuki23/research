import numpy as np
import itertools
from lib.voxel import coordinate_to_voxel_index, coordinate_to_grid

DELTA_INDEX = 1

def voxelizer_atom(
        atomic_symbols, 
        atom_coordinates,
        # atoms_amino,
        grid_origin,
        grid_dims,
        half_length_index_cutoff=5,
        length_voxel=0.5,
        factor=12,
        radiuses={'C': 1.69984, 'N': 1.62500, 'O': 1.51369, 'S': 1.78180, 'H': 1.2, 'B' : 1.92, 'F' : 1.47, 'P' : 1.80, 'I' : 1.98},
        dtype=np.float64
        ):

    atomic_symbol2index={index: i for i, index in enumerate(radiuses.keys())}
    diff_index_each_axis = range(-half_length_index_cutoff, half_length_index_cutoff + 1, 1)
    #itertools.product 使うと[001, 002, 003, ..., 998, 999]みたいに入る　
    diff_index = np.array([row for row in itertools.product(diff_index_each_axis, 
                                                            diff_index_each_axis, 
                                                            diff_index_each_axis)]).astype(np.int64)
    #各軸の両端にcutoff値だけpadding領域を追加
    lengths_index_voxel_pad = grid_dims + half_length_index_cutoff * 2
    #shape:(5,x_max-x_min+14*2, y_max-y_min+14*2, y_max-y_min+14*2)
    # *はリストのアンパック
    voxel_pad = np.zeros((len(radiuses), *lengths_index_voxel_pad))  
    for i in range(len(atomic_symbols)):
        atomic_symbol = atomic_symbols[i]
        atom_xyz = atom_coordinates[i]
        # atom_amino = atoms_amino[i]
        # if atom_amino == "LIGAND":
        #     continue

        # 各原子のボクセル番号
        atom_grid = coordinate_to_voxel_index(atom_xyz, grid_origin, length_voxel)
        # 原子のボクセル番号がボクセルの範囲内に収まっているかどうか
        skip = False
        for grid, length in zip(atom_grid, grid_dims):
            if grid > length:
                skip = True
                break 
        if skip:
            continue 

        atom_diff_grid_float = coordinate_to_grid(atom_xyz, grid_origin, length_voxel) % DELTA_INDEX

        distances = length_voxel * np.linalg.norm(
                                        diff_index - atom_diff_grid_float, 
                                        axis=1
                                        ).reshape(half_length_index_cutoff * 2 + 1, 
                                                half_length_index_cutoff * 2 + 1, 
                                                half_length_index_cutoff * 2 + 1)
    
        voxel_pad[
            atomic_symbol2index[atomic_symbol],
            atom_grid[0]: atom_grid[0] + half_length_index_cutoff * 2 + 1,
            atom_grid[1]: atom_grid[1] + half_length_index_cutoff * 2 + 1,
            atom_grid[2]: atom_grid[2] + half_length_index_cutoff * 2 + 1
                ] += 1 - np.exp(-((radiuses[atomic_symbol] / distances) ** factor))

    return voxel_pad[:, 
                    half_length_index_cutoff: -half_length_index_cutoff, 
                    half_length_index_cutoff: -half_length_index_cutoff, 
                    half_length_index_cutoff: -half_length_index_cutoff].astype(dtype)

