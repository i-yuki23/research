from WaterClassifier.LigandPocketDefinerOriginal import LigandPocketDefinerOriginal

class LigandPocketDefinerFactory:
    def __init__(self):
        self._rules = {
            "LigandPocketDefinerOriginal": LigandPocketDefinerOriginal,
        }

    def get_ligand_pocket_definer(self, definer_name, *args):
        definer_class = self._rules.get(definer_name)
        if definer_class is not None:
            return definer_class(*args)
        else:
            raise ValueError(f"Unknown definer: {definer_name}")