"""
holo状態のリガンドの露出表面積の割合を計算する
訓練データの絞り込みに使う→露出表面積が極端に小さいものはapo状態ではあり得ないので除外する
"""



import sys
sys.path.append('../..')

import freesasa
from lib.path import get_protein_path, get_ligand_path
from lib.pdb import get_pdb_names_by_txt
import numpy as np
from typing import Tuple

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
            return comp_ligand_sasa / ligand_sasa
        except Exception as e:
            raise ValueError(f"Error calculating ligand SASA for {self.complex_path}: {e}")

# リガンドの残基名（例: 'LIG'）
ligand_resname = 'LIG'
pdb_names = get_pdb_names_by_txt('/mnt/ito/pdbbind_raw/general_set/index/monomer_general_protein.txt')

ligand_sasa_list = []
for pdb_name in pdb_names:
    # pdb_name = '4lkk'
    print(pdb_name)
    sasa_calculator = LigandSASACalculator(
        f'/mnt/ito/pdbbind_raw/general_set/{pdb_name}/{pdb_name}_complex.pdb',
        f'/mnt/ito/pdbbind_raw/general_set/{pdb_name}/{pdb_name}_ligand.pdb',
        ligand_resname
    )
    ligand_sasa = sasa_calculator.calculate_ligand_sasa()
    ligand_sasa_list.append(ligand_sasa)
    print(ligand_sasa)
    # break

np.save('ligand_sasa_general.npy', np.array(ligand_sasa_list))
