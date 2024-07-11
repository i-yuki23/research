import sys
sys.path.append('..')

from Bio.PDB import PDBParser
import freesasa
from lib.voxel import coordinate_to_voxel_index, get_voxel_info
from lib.path import get_protein_path
from lib.pdb import get_all_pdb_names
import numpy as np
from preprocess.Class.WaterClassifier.LigandPocketDefinerOriginal import LigandPocketDefinerOriginal

class HydrophobicityCalculator:
    HYDROPHOBICITY_INDICES = {
        'ALA': 1.8, 'CYS': 2.5, 'ASP': -3.5, 'GLU': -3.5, 'PHE': 2.8, 'GLY': -0.4,
        'HIS': -3.2, 'ILE': 4.5, 'LYS': -3.9, 'LEU': 3.8, 'MET': 1.9, 'ASN': -3.5,
        'PRO': -1.6, 'GLN': -3.5, 'ARG': -4.5, 'SER': -0.8, 'THR': -0.7, 'VAL': 4.2,
        'TRP': -0.9, 'TYR': -1.3
    }

    def __init__(self, pdb_path, grid_origin, ligand_pocket_definer):
        self.pdb_path = pdb_path
        self.grid_origin = grid_origin
        self.ligand_pocket_definer = ligand_pocket_definer
        self._parse_structure(pdb_path)

    def _parse_structure(self, pdb_path):
        parser = PDBParser()
        self.structure = parser.get_structure('structure', pdb_path)
        fs_structure = freesasa.Structure(pdb_path)
        self.sasa_result = freesasa.calc(fs_structure)
        self.atom_indices = self._map_atom_indices()

    def _map_atom_indices(self):
        atom_indices = {}
        index = 0
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_indices[atom.get_serial_number()] = index
                        index += 1
        return atom_indices

    def _get_ligand_pocket(self):
        return self.ligand_pocket_definer.define_ligand_pocket()
    
    def _get_atoms_in_ligand_pocket(self, ligand_pocket_grid):
        atoms_in_ligand_pocket = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if self._is_atom_in_ligand_pocket(atom, ligand_pocket_grid):
                            atoms_in_ligand_pocket.append(atom)
        return atoms_in_ligand_pocket

    def _is_atom_in_ligand_pocket(self, atom, ligand_pocket_grid):
        x, y, z = coordinate_to_voxel_index(atom.get_coord(), self.grid_origin)
        range_to_check = 5
    
        x_min, x_max = max(0, x - range_to_check), min(ligand_pocket_grid.shape[0], x + range_to_check + 1)
        y_min, y_max = max(0, y - range_to_check), min(ligand_pocket_grid.shape[1], y + range_to_check + 1)
        z_min, z_max = max(0, z - range_to_check), min(ligand_pocket_grid.shape[2], z + range_to_check + 1)
    
        # Extract the subgrid around the point (x, y, z)
        subgrid = ligand_pocket_grid[x_min:x_max, y_min:y_max, z_min:z_max]
        if np.any(subgrid == 1):
            return True
    
        return False

    def calculate_hydrophobicity(self):
        ligand_pocket_grid = self._get_ligand_pocket()
        atoms_in_ligand_pocket = self._get_atoms_in_ligand_pocket(ligand_pocket_grid)

        total_hydrophobicity = 0
        total_sasa = 0
        for atom in atoms_in_ligand_pocket:
            try:
                hydrophobicity, sasa = self._calculate_atom_hydrophobicity(atom)
                total_hydrophobicity += hydrophobicity
                total_sasa += sasa
            except (KeyError, AssertionError) as e:
                continue
            
        if total_sasa == 0:
            return 0
        return total_hydrophobicity / total_sasa

    def _calculate_atom_hydrophobicity(self, atom):
        atom_id = atom.get_serial_number()
        if atom_id not in self.atom_indices:
            raise KeyError(f"Atom serial number {atom_id} not found in atom_indices.")
        
        sasa_index = self.atom_indices[atom_id]

        sasa = self.sasa_result.atomArea(sasa_index)
        hydrophobicity = self.HYDROPHOBICITY_INDICES.get(atom.get_parent().resname, 0)
        return hydrophobicity * sasa, sasa


hydrophobicities = []
for pdb_name in get_all_pdb_names():
    grid_dims, grid_origin = get_voxel_info(pdb_name)
    ligand_pocket_definer = LigandPocketDefinerOriginal(pdb_name, grid_dims, grid_origin, 8)
    hydrophobicity_calculator = HydrophobicityCalculator(get_protein_path(pdb_name), grid_origin, ligand_pocket_definer)
    hydrophobicities.append(hydrophobicity_calculator.calculate_hydrophobicity())

np.save('hydrophobicities_std.npy', np.array(hydrophobicities))
