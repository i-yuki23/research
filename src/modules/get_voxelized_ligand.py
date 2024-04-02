from lib.path import get_ligand_path
from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb
from modules.voxelizer import voxelizer_atom

def get_voxelized_ligand(pdb_name, grid_dims, grid_origin):

    ligand_path = get_ligand_path(pdb_name)
    ligand_coordinates = get_coordinates_from_pdb(ligand_path)
    ligand_atomic_symbols = get_atomic_symbols_from_pdb(ligand_path)

    ligand_voxelized = voxelizer_atom(
        atomic_symbols=ligand_atomic_symbols,
        atom_coordinates=ligand_coordinates,
        grid_origin=grid_origin,
        grid_dims=grid_dims,
    )

    return ligand_voxelized