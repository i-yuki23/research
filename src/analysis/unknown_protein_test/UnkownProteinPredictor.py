class UnkownProteinPredictor:
    def __init__(self, test_data_loader, prediction_executor, prediction_analyzer):
        self.test_data_loader = test_data_loader
        self.prediction_executor = prediction_executor
        self.prediction_analyzer = prediction_analyzer

    def run(self):
        test_data, test_water_ids = self.test_data_loader.get_test_data_and_water_ids()
        
        prediction = self.prediction_executor.predict(test_data)
        
        self.prediction_analyzer.set_prediction_and_predicted_labels(prediction)
        self.prediction_analyzer.save_prediction_results_pdb(test_water_ids)
