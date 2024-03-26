from lib.path import get_water_path
from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb
from lib.voxel import read_xyzv
from modules.voxelizer import voxelizer_atom

WATER_CHANNEL_INDEX = 2

def get_voxelized_water(pdb_name):

    water_path = get_water_path(pdb_name)
    water_coordinates = get_coordinates_from_pdb(water_path)
    water_atomic_symbols = get_atomic_symbols_from_pdb(water_path)

    _, grid_dims, grid_origin = read_xyzv(pdb_name)

    water_voxelized = voxelizer_atom(
        atomic_symbols=water_atomic_symbols,
        atom_coordinates=water_coordinates,
        grid_origin=grid_origin,
        grid_dims=grid_dims,
    )

    return water_voxelized[WATER_CHANNEL_INDEX]

