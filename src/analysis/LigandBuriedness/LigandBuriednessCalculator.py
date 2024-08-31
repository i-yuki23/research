import sys
sys.path.append('../..')

import freesasa
from lib.path import get_protein_path, get_ligand_path
from lib.pdb import get_all_pdb_names
import numpy as np
from typing import Tuple

class LigandBuriednessCalculator:
    def __init__(self, pdb_path: str, ligand_path: str, complex_path: str):
        self.pdb_path = pdb_path
        self.ligand_path = ligand_path
        self.complex_path = complex_path
        
        self.protein_sasa, self.ligand_sasa, self.complex_sasa = self._calculate_sasas()

    def _calculate_sasas(self) -> Tuple[float, float, float]:
        protein_sasa = self._get_sasa(self.pdb_path)
        ligand_sasa = self._get_sasa(self.ligand_path)
        complex_sasa = self._get_sasa(self.complex_path)

        return protein_sasa, ligand_sasa, complex_sasa

    def _get_sasa(self, path: str) -> float:
        try:
            structure = freesasa.Structure(path)
            sasa_result = freesasa.calc(structure)
            return sasa_result.totalArea()
        except Exception as e:
            raise ValueError(f"Error calculating SASA for {path}: {e}")

    def calculate_buriedness(self) -> float:
        buriedness = (self.ligand_sasa + self.protein_sasa - self.complex_sasa) / self.ligand_sasa
        return buriedness



ligand_buriedness_list = []
for pdb_name in get_all_pdb_names():
    averageSASA_calculator = LigandBuriednessCalculator(get_protein_path(pdb_name), get_ligand_path(pdb_name), f'../../../data/protein_ligand_complex/{pdb_name}/{pdb_name}_complex.pdb')
    ave_sasa_list.append(averageSASA_calculator.calculate_buriedness())
    print(ave_sasa_list)
    break

# np.save('ave_sasa.npy', np.array(ave_sasa_list))
