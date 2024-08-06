# Ligand Pocketの平均SASAを計算する

import sys
sys.path.append('../..')

from Bio.PDB import PDBParser
import freesasa
from lib.voxel import coordinate_to_voxel_index, get_voxel_info
from lib.path import get_protein_path
from lib.pdb import get_all_pdb_names
import numpy as np
from preprocess.Class.WaterClassifier.LigandPocketDefinerOriginal import LigandPocketDefinerOriginal

class AverageSASACalculator:

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

    def calculate_ligand_pocket_ave_sasa(self):
        ligand_pocket_grid = self._get_ligand_pocket()
        atoms_in_ligand_pocket = self._get_atoms_in_ligand_pocket(ligand_pocket_grid)

        sasa_list = []
        for atom in atoms_in_ligand_pocket:
            try:
                sasa_list.append(self._calculate_atom_sasa(atom))
            except (KeyError, AssertionError) as e:
                continue
            
        sasa_len = len(sasa_list)
        if sasa_len == 0:
            return 0
        return sum(sasa_list) / sasa_len

    def _calculate_atom_sasa(self, atom):
        atom_id = atom.get_serial_number()
        if atom_id not in self.atom_indices:
            raise KeyError(f"Atom serial number {atom_id} not found in atom_indices.")
        
        sasa_index = self.atom_indices[atom_id]

        return self.sasa_result.atomArea(sasa_index)


ave_sasa_list = []
for pdb_name in get_all_pdb_names():
    grid_dims, grid_origin = get_voxel_info(pdb_name)
    ligand_pocket_definer = LigandPocketDefinerOriginal(pdb_name, grid_dims, grid_origin, 8)
    averageSASA_calculator = AverageSASACalculator(get_protein_path(pdb_name), grid_origin, ligand_pocket_definer)
    ave_sasa_list.append(averageSASA_calculator.calculate_ligand_pocket_ave_sasa())

np.save('ave_sasa.npy', np.array(ave_sasa_list))
