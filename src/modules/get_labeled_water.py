import numpy as np
from lib.pdb import get_coordinates_from_pdb, get_atom_id_from_pdb
from lib.path import get_displaceable_water_path, get_non_displaceable_water_path

def get_displaceable_water_coords(pdb_name: str, ligand_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str) -> np.ndarray:
    """
    Extracts coordinates of displaceable water molecules from a PDB file.
    
    Parameters:
    - pdb_name: str, name of the PDB file
    
    Returns:
    - np.ndarray, coordinates of displaceable water molecules
    """
    displaceable_water_path = get_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    try:
        displaceable_water_coords = get_coordinates_from_pdb(displaceable_water_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Displaceable water file for {pdb_name} does not exist.")
    
    return displaceable_water_coords
    
def get_non_displaceable_water_coords(pdb_name: str, ligand_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str) -> np.ndarray:
    
    non_displaceable_water_path = get_non_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    try:
        non_displaceable_water_coords = get_coordinates_from_pdb(non_displaceable_water_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Non-displaceable water file for {pdb_name} does not exist.")

    return non_displaceable_water_coords    

def get_displaceable_water_ids(pdb_name: str, ligand_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str) -> np.ndarray:
    displaceable_water_path = get_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    try:
        displaceable_water_ids = get_atom_id_from_pdb(displaceable_water_path, 'ATOM')
    except FileNotFoundError:
        raise FileNotFoundError(f"Displaceable water file for {pdb_name} does not exist.")
    
    return displaceable_water_ids

def get_non_displaceable_water_ids(pdb_name: str, ligand_voxel_num: int, classifying_rule: str, ligand_pocket_definer: str) -> np.ndarray:
    non_displaceable_water_path = get_non_displaceable_water_path(pdb_name, ligand_voxel_num, classifying_rule, ligand_pocket_definer)
    try:
        non_displaceable_water_ids = get_atom_id_from_pdb(non_displaceable_water_path, 'ATOM')
    except FileNotFoundError:
        raise FileNotFoundError(f"Displaceable water file for {pdb_name} does not exist.")
    
    return non_displaceable_water_ids