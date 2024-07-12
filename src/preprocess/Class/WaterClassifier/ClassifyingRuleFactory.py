from Class.WaterClassifier.WaterClassifyingRuleSurface import WaterClassifyingRuleSurface
from Class.WaterClassifier.WaterClassifyingRuleCenter import WaterClassifyingRuleCenter
from Class.WaterClassifier.WaterClassifyingRuleEmbedding import WaterClassifyingRuleEmbedding


class ClassifyingRuleFactory:
    def __init__(self):
        self._rules = {
            "WaterClassifyingRuleSurface": WaterClassifyingRuleSurface,
            "WaterClassifyingRuleCenter": WaterClassifyingRuleCenter,
            "WaterClassifyingRuleEmbedding": WaterClassifyingRuleEmbedding,
        }

    def get_rule(self, rule_name, *args):
        rule_class = self._rules.get(rule_name)
        if rule_class is not None:
            return rule_class(*args)
        else:
            raise ValueError(f"Unknown rule: {rule_name}")