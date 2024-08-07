import numpy as np
import os
from lib.path import get_nas_data_dir
from lib.helper import make_dir
from collections import defaultdict

# Constants for readability
ATOM_COORD_START = 30
ATOM_COORD_END = 54
ATOMIC_ID_START = 7
ATOMIC_ID_END = 11
ATOMIC_SYMBOL_POS_START = 76
ATOMIC_SYMBOL_POS_END = 78
CHAIN_NAME_POS = 21

def get_coords(line):
    return [float(line[ATOM_COORD_START:ATOM_COORD_START+8].strip()),        
                              float(line[ATOM_COORD_START+8:ATOM_COORD_START+16].strip()),
                              float(line[ATOM_COORD_START+16:ATOM_COORD_END].strip())]

def get_coordinates_from_pdb(path_to_pdb, exclude_hydrogens=True, type="ATOM"):
    """
    Extracts coordinates of atoms from a PDB file.
    
    Parameters:
    - path_to_pdb: str, path to the PDB file
    - exclude_hydrogens: bool, whether to exclude hydrogen atoms
    
    Returns:
    - np.ndarray, coordinates of atoms
    """
    if not os.path.exists(path_to_pdb):
        raise FileNotFoundError(f"{path_to_pdb} does not exist.")
    
    coords_list = []
    with open(path_to_pdb, 'r') as f:
        for line in f:
            if line.startswith(type):
                if type == "ATOM" and exclude_hydrogens and line[ATOMIC_SYMBOL_POS_START:ATOMIC_SYMBOL_POS_END].strip() == 'H':
                    continue
                coords = get_coords(line)
                coords_list.append(coords)
                
    return np.array(coords_list, dtype="float64")

def get_chains_coordinates(path_to_pdb, exclude_hydrogens=True, type="ATOM"):
    chains_coords_list = []
    coords_list = []
    chain_name = None
    with open(path_to_pdb, 'r') as f:
        for line in f:
            if line.startswith(type):
                if exclude_hydrogens and line[ATOMIC_SYMBOL_POS_START:ATOMIC_SYMBOL_POS_END].strip() == 'H':
                    continue
                current_chain_name = line[CHAIN_NAME_POS]
                if chain_name is None:
                    chain_name = current_chain_name
                if current_chain_name == chain_name:
                    coords_list.append(get_coords(line))
                else:
                    chains_coords_list.append(np.array(coords_list))
                    coords_list = [get_coords(line)]
                    chain_name = current_chain_name
        if coords_list:
            chains_coords_list.append(np.array(coords_list))
                
    return chains_coords_list

def get_ligand_coordinates_from_pdb(path_to_pdb):
    """
    Extracts coordinates of ligand atoms from a PDB file.
    
    Parameters:
    - path_to_pdb: str, path to the PDB file
    
    Returns:
    - np.ndarray, coordinates of atoms
    """
    if not os.path.exists(path_to_pdb):
        raise FileNotFoundError(f"{path_to_pdb} does not exist.")
    
    coords_list = []
    with open(path_to_pdb, 'r') as f:
        for line in f:
            if line.startswith("TER"):
                break
            if line.startswith("ATOM") or line.startswith("HETATM"):
                coords = get_coords(line)
                coords_list.append(coords)
                
    return np.array(coords_list, dtype="float64")

def get_atomic_symbols_from_pdb(path_to_pdb, exclude_hydrogens=True):
    """
    Extracts atomic symbols from a PDB file.
    
    Parameters:
    - path_to_pdb: str, path to the PDB file
    - exclude_hydrogens: bool, whether to exclude hydrogen atoms
    
    Returns:
    - np.ndarray, atomic symbols
    """
    if not os.path.exists(path_to_pdb):
        raise FileNotFoundError(f"{path_to_pdb} does not exist.")
    
    atomic_symbols = []
    with open(path_to_pdb, 'r') as f:
        for line in f:
            if line.startswith("TER"):
                break
            if line.startswith("ATOM"):
                symbol = line[ATOMIC_SYMBOL_POS_START:ATOMIC_SYMBOL_POS_END].strip()
                if exclude_hydrogens and symbol == 'H':
                    continue
                if symbol == 'A':
                    symbol = 'C'
                if symbol == 'D':
                    symbol = 'O'
                atomic_symbols.append(symbol)

    return np.array(atomic_symbols)

def get_atom_id_from_pdb(path_to_pdb, type):
    """
    Extracts atomic ids from a PDB file.
    
    Parameters:
    - path_to_pdb: str, path to the PDB file
    
    Returns:
    - np.ndarray, atomic ids
    """
    if not os.path.exists(path_to_pdb):
        raise FileNotFoundError(f"{path_to_pdb} does not exist.")
    
    atomic_ids = []
    with open(path_to_pdb, 'r') as f:
        for line in f:
            if line.startswith("TER"):
                break
            if line.startswith(type):
                atomic_id = line[ATOMIC_ID_START:ATOMIC_ID_END+1].strip() 
                atomic_ids.append(atomic_id)
    return np.array(atomic_ids)

def filter_atoms_and_create_new_pdb(input_pdb_path: str, output_pdb_path: str, target_atom_ids: list, type="ATOM") -> None:
    make_dir(output_pdb_path)
    atom_dict = {}
    with open(input_pdb_path, 'r') as f:
        for line in f:
            if line.startswith("TER"):
                break
            if line.startswith(type):    
                atom_id = line[ATOMIC_ID_START:ATOMIC_ID_END+1].strip() 
                atom_dict[int(atom_id)] = line
                
    with open(output_pdb_path, 'w') as f:
        for target_atom_id in target_atom_ids:
            if target_atom_id in atom_dict:
                f.write(atom_dict[target_atom_id])

def get_all_pdb_names():
    dir_path = get_nas_data_dir()
    return [pdb_name for pdb_name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, pdb_name))]

def get_pdb_names_by_txt(txt_path):
    with open(txt_path, 'r') as f:
        return f.read().splitlines()

def get_clusters_from_pdb(path_to_pdb, type="HETATM"):
    """
    Extracts coordinates of atoms from each cluster in a PDB file.
    
    Parameters:
    - path_to_pdb: str, path to the PDB file
    - exclude_hydrogens: bool, whether to exclude hydrogen atoms
    - type: str, type of atoms to extract ("ATOM" or "HETATM")
    
    Returns:
    - list of np.ndarray, list containing the coordinates of atoms for each cluster
    """
    if not os.path.exists(path_to_pdb):
        raise FileNotFoundError(f"{path_to_pdb} does not exist.")
    
    clusters = []
    current_cluster = []

    with open(path_to_pdb, 'r') as f:
        for line in f:
            if line.startswith("TER") or line.startswith("ENDMDL"):
                if current_cluster:  # Ensure there's data to add
                    clusters.append(np.array(current_cluster, dtype="float64"))
                    current_cluster = []
            elif line.startswith(type):
                coords = get_coords(line)
                current_cluster.append(coords)
    
    # Add the last cluster if not ended with TER
    if current_cluster:
        clusters.append(np.array(current_cluster, dtype="float64"))
                
    return clusters

def get_atoms_coords_for_each_atom_type(path_to_pdb: str) -> dict:
    """
    Extracts coordinates of atoms from a PDB file for each atom type.

    Args:
        pdb_file (str): Path to a PDB file.

    Returns:
        dict: key -> element, value -> list of coordinates for corresponding element
    """
    if not os.path.exists(path_to_pdb):
        raise FileNotFoundError(f"{path_to_pdb} does not exist.")
    
    atoms_coords_for_each_atom_type = defaultdict(list)
    with open(path_to_pdb, 'r') as f:
        for line in f:
            if line.startswith("TER"):
                break
            if line.startswith("ATOM"):
                coords = get_coords(line)
                element = line[ATOMIC_SYMBOL_POS_START:ATOMIC_SYMBOL_POS_END].strip()
                if element == 'A':
                    element = 'C'
                if element == 'D':
                    element = 'O'  
                atoms_coords_for_each_atom_type[element].append(coords)

    # Convert lists to 2D numpy arrays
    for element in atoms_coords_for_each_atom_type:
        atoms_coords_for_each_atom_type[element] = np.array(atoms_coords_for_each_atom_type[element])       
    
    return atoms_coords_for_each_atom_type