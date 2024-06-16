class UnkownProteinPredictor:
    def __init__(self, test_data_loader, prediction_executor, prediction_result_saver):
        self.test_data_loader = test_data_loader
        self.prediction_executor = prediction_executor
        self.prediction_result_saver = prediction_result_saver

    def run(self):
        test_data = self.test_data_loader.load_data()
        
        # 5. Execute predictions on test data
        prediction = self.prediction_executor.predict(test_data)
        
        # 6. Save prediction results as PDB files
        self.prediction_analyzer.save(prediction)

# Example usage
test_data_loader = TestDataLoader()
prediction_executor = PredictionExecutor(model)
prediction_result_saver = PredictionAnalyzer()

prediction_pipeline = PredictionPipeline(
    test_data_loader=test_data_loader,
    prediction_executor=prediction_executor,
    prediction_result_saver=prediction_result_saver
)

prediction_pipeline.run()
