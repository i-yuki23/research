import freesasa

class ResSASACalculator:
    def __init__(self, pdb_path: str, resname: str):
        self.pdb_path = pdb_path
        self.resname = resname

    def _get_sasa(self, path: str) -> float:
        try:
            structure = freesasa.Structure(path)
            sasa_result = freesasa.calc(structure)
            return structure, sasa_result
        except Exception as e:
            raise ValueError(f"Error calculating SASA for {path}: {e}")

    def calculate_res_sasa(self) -> float:
        structure, sasa_result = self._get_sasa(self.pdb_path)
        # 指定した残機のSASAを選択する
        selection = freesasa.selectArea([f"water, resn {self.resname}"], structure, sasa_result)
        res_sasa = selection['water']
        return res_sasa
