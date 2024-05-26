from WaterClassifier.LigandPocketDefinerOriginal import LigandPocketDefinerOriginal
from WaterClassifier.LigandPocketDefinerGhecom import LigandPocketDefinerGhecom

class LigandPocketDefinerFactory:
    def __init__(self):
        self._rules = {
            "LigandPocketDefinerOriginal": LigandPocketDefinerOriginal,
            "LigandPocketDefinerGhecom": LigandPocketDefinerGhecom
        }

    def get_ligand_pocket_definer(self, definer_name, *args):
        definer_class = self._rules.get(definer_name)
        print(definer_name)
        if definer_class is not None:
            return definer_class(*args)
        else:
            raise ValueError(f"Unknown definer: {definer_name}")