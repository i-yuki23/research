import sys 
sys.path.append('..')
from lib.path import get_protein_path
from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb
from lib.voxel import get_voxel_info
from modules.voxelizer import voxelizer_atom

def get_voxelized_protein(pdb_name):

    protein_path = get_protein_path(pdb_name)
    protein_coordinates = get_coordinates_from_pdb(protein_path)
    protein_atomic_symbols = get_atomic_symbols_from_pdb(protein_path)
    grid_dims, grid_origin = get_voxel_info(pdb_name)

    protein_voxelized = voxelizer_atom(
        atomic_symbols=protein_atomic_symbols,
        atom_coordinates=protein_coordinates,
        grid_origin=grid_origin,
        grid_dims=grid_dims,
    )

    return protein_voxelized[:5]      # 5 channels for protein voxelization

print(get_voxelized_protein('4lkk').shape)