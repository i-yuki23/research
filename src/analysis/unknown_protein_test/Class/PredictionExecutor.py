class PredictionExecutor:

    def __init__(self, model, checkpoint_path):
        self.model = model
        self.model.load_weights(checkpoint_path)

    def predict(self, test_data):
        prediction = self.model.predict(test_data)
        return prediction