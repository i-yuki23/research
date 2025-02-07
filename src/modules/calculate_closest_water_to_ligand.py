import sys 
sys.path.append("..")

import os
from lib.path import get_all_pdb_bind_dir, get_ligand_path
from lib.pdb import get_coordinates_from_pdb, get_atoms_coords_for_each_atom_type
from lib.helper import calculate_closest_distance
from collections import defaultdict

def calculate_closet_water_distance_to_ligand_for_each_element(pdb_names: list) ->dict:
    """
    Calculate the closest distance between each element in the ligand and the closest water molecule.

    Args:
        pdb_names (list): list of pdb names.
        
    Returns:
        dict: key -> element, value -> a list of closest distance between the element and the closest water molecule for all the proteins.
    """
    closest_distance_for_each_element = defaultdict(list)
    pdb_names_for_each_element = defaultdict(list)
    all_pdb_bind_dir = get_all_pdb_bind_dir()
    for pdb_name in pdb_names:
        crystal_water_path = os.path.join(all_pdb_bind_dir, f'{pdb_name}/{pdb_name}_HOH.pdb')
        crystal_water_coordinates = get_coordinates_from_pdb(crystal_water_path, type='HETATM')
        if crystal_water_coordinates.size == 0:
            continue
        ligand_path = os.path.join(all_pdb_bind_dir, f'{pdb_name}/{pdb_name}_ligand.pdb')
        atoms_coords_for_each_atom_type = get_atoms_coords_for_each_atom_type(ligand_path)
    
        for element in atoms_coords_for_each_atom_type:
            closest_distance_for_each_element[element].append(calculate_closest_distance(atoms_coords_for_each_atom_type[element], crystal_water_coordinates))
            pdb_names_for_each_element[element].append(pdb_name)
    return closest_distance_for_each_element, pdb_names_for_each_element


# print(calculate_closet_water_distance_to_ligand_for_each_element(['2bdj']))