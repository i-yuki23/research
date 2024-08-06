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

    def __init__(self, pdb_path):
        self.pdb_path = pdb_path
        self._parse_structure(pdb_path)

    def _parse_structure(self, pdb_path):
        parser = PDBParser()
        self.structure = parser.get_structure('structure', pdb_path)
        fs_structure = freesasa.Structure(pdb_path)
        self.sasa_result = freesasa.calc(fs_structure)
        print(self.sasa_result)
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

    
    def _get_ligand_atoms(self):
        ligand_atoms = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.resname == 'SIA':
                        for atom in residue:
                            ligand_atoms.append(atom)
        print(ligand_atoms)

        return ligand_atoms

    def calculate_ligand_sasa(self):
        ligand_atoms = self._get_ligand_atoms()

        ligand_sasa = 0
        for atom in ligand_atoms:
            try:
                ligand_sasa += self._calculate_atom_sasa(atom)
            except (KeyError, AssertionError) as e:
                continue

        return ligand_sasa

    def _calculate_atom_sasa(self, atom):
        atom_id = atom.get_serial_number()
        if atom_id not in self.atom_indices:
            raise KeyError(f"Atom serial number {atom_id} not found in atom_indices.")
        
        sasa_index = self.atom_indices[atom_id]
        print(self.sasa_result.atomArea(4100))
        print(sasa_index)
        return self.sasa_result.atomArea(sasa_index)


ave_sasa_list = []
for pdb_name in get_all_pdb_names():
    pdb_name = '4lkk'
    protein_path = f'../../../data/protein_ligand_complex/{pdb_name}/{pdb_name}_complex.pdb'
    averageSASA_calculator = AverageSASACalculator(protein_path)
    ave_sasa_list.append(averageSASA_calculator.calculate_ligand_sasa())
    print(ave_sasa_list)
    break

np.save('ligand_sasa.npy', np.array(ave_sasa_list))
