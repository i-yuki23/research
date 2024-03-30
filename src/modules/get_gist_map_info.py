from lib.path import get_gist_path
from lib.dx import read_dx
from typing import Tuple
import numpy as np

def get_gist_map_info(pdb_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    gist_path = get_gist_path(pdb_name)
    gist_map, grid_dims, grid_origin = read_dx(gist_path)

    return gist_map, grid_dims, grid_origin