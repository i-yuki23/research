from WaterClassifier.WaterClassifyingRuleSurface import WaterClassifyingRuleSurface
from WaterClassifier.WaterClassifyingRuleCenter import WaterClassifyingRuleCenter

class ClassifyingRuleFactory:
    def __init__(self):
        self._rules = {
            "WaterClassifyingRuleSurface": WaterClassifyingRuleSurface,
            "WaterClassifyingRuleCenter": WaterClassifyingRuleCenter,
        }

    def get_rule(self, rule_name, *args):
        rule_class = self._rules.get(rule_name)
        if rule_class is not None:
            return rule_class(*args)
        else:
            raise ValueError(f"Unknown rule: {rule_name}")