from Bio.PDB import PDBParser
from collections import defaultdict
import numpy as np

def get_residue_info(pdb_file_path: str) -> np.ndarray:
    """
    Extract residue information from a PDB file. 

    Args:
        pdb_file_path (str): Path to a PDB file.
    
    Returns:
        np.ndarray: A 1D array containing residue names.
    """
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file_path)
    
    residues_list = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_name = residue.get_resname()
                
                for atom in residue:
                    residues_list.append(residue_name)
    
    return np.array(residues_list)

def get_atom_names(pdb_file: str) -> np.ndarray:
    """
    Get the atom names in a PDB file. (Not an element name, but a specific atom name like "CA" or "CB")

    Args:
        pdb_file (str): Path to a PDB file.

    Returns:
        np.ndarray: A 1D array containing atom names.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein_structure", pdb_file)

    atom_names = []
    atoms = structure.get_atoms()
    for atom in atoms:
        atom_names.append(atom.get_name())
    
    return np.array(atom_names)






