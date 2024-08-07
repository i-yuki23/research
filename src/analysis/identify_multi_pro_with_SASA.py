import sys
sys.path.append('..')

import freesasa
from lib.path import get_ligand_path
from lib.pdb import get_all_pdb_names
import numpy as np
import os
import glob

class LigandSASACalculator:
    def __init__(self, complex_path: str, ligand_path: str, ligand_resname: str):
        self.complex_path = complex_path
        self.ligand_path = ligand_path
        self.ligand_resname = ligand_resname

    def _get_sasa(self, path: str) -> float:
        try:
            structure = freesasa.Structure(path)
            sasa_result = freesasa.calc(structure)
            return structure, sasa_result
        except Exception as e:
            raise ValueError(f"Error calculating SASA for {path}: {e}")

    def calculate_ligand_sasa(self) -> float:
        try:
            structure_comp, sasa_result_comp = self._get_sasa(self.complex_path)
            _, ligand_sasa_result = self._get_sasa(self.ligand_path)
            ligand_sasa = ligand_sasa_result.totalArea()

            # リガンドのSASAを選択して計算する
            selection = freesasa.selectArea([f"ligand, resn {self.ligand_resname}"], structure_comp, sasa_result_comp)
            comp_ligand_sasa = selection['ligand']
            return ligand_sasa - comp_ligand_sasa
        except Exception as e:
            raise ValueError(f"Error calculating ligand SASA for {self.complex_path}: {e}")

# リガンドの残基名（例: 'LIG'）
ligand_resname = 'LIG'

data_dir = "../../data/protein_ligand_complex/"

valid_protein_list = []
non_zero_sasa_num = 0
for pdb_name in get_all_pdb_names():
    print(pdb_name)
    chain_path = os.path.join(data_dir, f'{pdb_name}/chain_*.pdb')
    chain_path_list = glob.glob(chain_path)
    for chain_path in chain_path_list:

        sasa_calculator = LigandSASACalculator(
            chain_path,
            f'/home/ito/research/data/protein_ligand_complex/{pdb_name}/{pdb_name}_ligand.pdb',
            ligand_resname
        )
        ligand_sasa_diff = sasa_calculator.calculate_ligand_sasa()
        print(ligand_sasa_diff)
        if ligand_sasa_diff != 0:
            non_zero_sasa_num += 1
        if non_zero_sasa_num == 2:
            break
    if non_zero_sasa_num == 1:
        valid_protein_list.append(pdb_name)

print(valid_protein_list)

with open('/home/ito/research/data/valid_protein.txt', 'w') as f:
    for protein in valid_protein_list:
        f.write(protein + '\n')