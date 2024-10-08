import pandas as pd
import numpy as np
import os
from lib.path import get_xyzv_path, get_gr_path

# Constants for readability
ATOM_COORD_START = 30
ATOM_COORD_END = 54
ATOMIC_SYMBOL_POS = 77

def coordinate_to_voxel_index(coordinate: np.ndarray, grid_origin: np.ndarray, length_voxel=0.5) -> np.ndarray:
    return ((coordinate - grid_origin) // length_voxel).astype(np.int64)

def coordinate_to_grid(coordinate, grid_origin, length_voxel):
    return (coordinate - grid_origin) / length_voxel

def read_xyzv(pdb_name):
    """
    Converts XYZV file data to a voxel representation.
    
    Parameters:
    - path_to_xyzv: str, path to the .xyzv file
    
    Returns:
    - voxel: np.ndarray, voxel representation of the volume data
    - grid: np.ndarray, dimensions of the voxel grid
    - init: np.ndarray, starting coordinate of the grid
    """
    path_to_xyzv = get_xyzv_path(pdb_name)
    if not os.path.exists(path_to_xyzv):
        raise FileNotFoundError(f"{path_to_xyzv} does not exist.")

    df = pd.read_csv(path_to_xyzv, header=None, delim_whitespace=True)
    x, y, z, v = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3].values
    
    grid_dims = [x.nunique(), y.nunique(), z.nunique()]
    voxel = v.reshape(*grid_dims, order='F')
    
    grid_origin = np.array([x.min(), y.min(), z.min()])
    return voxel, np.array(grid_dims), grid_origin

def get_voxel_info(pdb_name=None, data_type=None, dx_path=None):
    if pdb_name:
        if data_type is None:
            raise ValueError("pdb_nameが指定されている場合はdata_typeもセットする必要があります")
        voxel_path = get_gr_path(pdb_name, data_type)
    elif dx_path:
        voxel_path = dx_path
    else:
        raise ValueError("pdb_nameまたはdx_pathのどちらかを指定する必要があります")
    
    with open(voxel_path, 'r') as f:
        file = f.readlines()
    
    grid_dims = file[0].strip().split()[5:8]
    grid_origin = file[1].strip().split()[1:4]
    
    return np.array([int(grid_dims[i]) for i in range(3)]), np.array([float(grid_origin[i]) for i in range(3)])


def extract_surroundings_voxel(voxel_indices, grid_dims, voxel_num):
    """
    Extracts the surroundings of atoms within a voxel grid, allowing for voxel_num = 0.

    Parameters:
    - voxel_indices: np.ndarray, indices of atoms in the voxel grid
    - grid_dims: np.ndarray, dimensions of the voxel grid
    - voxel_num: int, size of the surrounding area to extract or 0 for the voxel itself

    Returns:
    - np.ndarray, voxel grid with surroundings marked
    """
    surroundings = np.zeros(grid_dims, dtype=int)
    for x, y, z in voxel_indices:
        if voxel_num == 0:
            # Directly mark the voxel if voxel_num is 0
            surroundings[x, y, z] = 1
        else:
            # Mark the surrounding area for voxel_num > 0
            surroundings[max(0, x-voxel_num):min(x+voxel_num+1, grid_dims[0]),
                            max(0, y-voxel_num):min(y+voxel_num+1, grid_dims[1]),
                            max(0, z-voxel_num):min(z+voxel_num+1, grid_dims[2])] = 1
    return surroundings