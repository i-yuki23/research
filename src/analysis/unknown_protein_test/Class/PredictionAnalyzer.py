import numpy as np
from lib.path import get_test_water_path
from lib.pdb import filter_atoms_and_create_new_pdb
from lib.helper import make_dir


class PredictionAnalyzer:

    def __init__(self, protein_name, apo_name, holo_name, output_dir, optimal_threshold=0.5):
        self.protein_name = protein_name
        self.apo_name = apo_name
        self.holo_name = holo_name
        self.output_dir = output_dir
        self.optimal_threshold = optimal_threshold

    def set_prediction_and_predicted_labels(self, prediction):
        self.prediction = prediction
        self.predicted_labels = self.__custom_threshold(prediction)

    def __custom_threshold(self, prediction):
        return (prediction > self.optimal_threshold).astype(int)
    
    def get_pos_and_neg_water_ids(self, test_water_ids):
        pos_water_ids = test_water_ids[np.where(self.predicted_labels == 1)[0]]
        neg_water_ids = test_water_ids[np.where(self.predicted_labels == 0)[0]]
        return pos_water_ids, neg_water_ids
    
    def save_prediction_results_pdb(self, test_water_ids):
        pos_water_ids, neg_water_ids = self.get_pos_and_neg_water_ids(test_water_ids)

        # Save and process each type of data
        labels = ["pos", "neg"]
        ids_list = [pos_water_ids, neg_water_ids]
        for label, water_ids in zip(labels, ids_list):
            output_path = self.output_dir + f"{label}_pred_O_placed_{self.apo_name}_3.0.pdb"
            input_path = get_test_water_path(self.protein_name, self.apo_name)
            # Process the predicted data to filter and create a new PDB for each label type
            make_dir(output_path)
            filter_atoms_and_create_new_pdb(input_path, output_path, water_ids)
