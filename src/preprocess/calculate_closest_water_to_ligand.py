import sys 
sys.path.append("..")

from lib.path import get_crystal_water_path, get_ligand_path, get_protein_path
from lib.pdb import get_coordinates_from_pdb, get_atoms_coords_for_each_atom_type


crystal_water_coordinates = get_coordinates_from_pdb(get_crystal_water_path("10gs"), type='HETATM')


print(get_atoms_coords_for_each_atom_type(get_protein_path('10gs')))


