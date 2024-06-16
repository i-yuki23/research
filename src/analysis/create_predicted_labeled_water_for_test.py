import sys
import os
sys.path.append('..')
from lib.path import get_training_data_dir, get_predicted_labeled_water_path, get_displaceable_water_path, get_non_displaceable_water_path
from lib.pdb import filter_atoms_and_create_new_pdb
from analysis.unknown_protein_test.Class.TestDataLoader import DataLoaderForTest
from data_loader.DoubleDataLoader import DoubleDataLoader
from models.LeNet import LeNet
from models.ResNet import ResNet
from lib.helper import get_latest_checkpoint
from tensorflow.keras.losses import BinaryCrossentropy
from custom_losses.dice import dice_loss, dice_coefficient
from tensorflow.keras.metrics import Recall, Precision
from sklearn.metrics import roc_curve
import numpy as np

def calculate_optimal_threshold(test_data_label, prediction_values):
    fpr, tpr, thresholds = roc_curve(test_data_label, prediction_values)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold_youden = thresholds[optimal_idx]
    return optimal_threshold_youden

def custom_threshold(prediction, threshold):
    return (prediction > threshold).astype(int)

# Define the parameters
pdb_name = '2h14'
holo_name = '3smrA'
DATA_TYPE1 = 'gr'
# DATA_TYPE2 = 'Protein'
DATA_TYPE = f"{DATA_TYPE1}_{DATA_TYPE2}" if 'DATA_TYPE2' in locals() else DATA_TYPE1
is_augmented = True

DATA_VOXEL_NUM = 10
CLASSIFYING_RULE = 'WaterClassifyingRuleSurface'
LIGAND_POCKET_DEFINER = 'LigandPocketDefinerOriginal'
LIGAND_VOXEL_NUM = 8


# Load the test data
training_data_dir1 = get_training_data_dir(DATA_TYPE1, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)
# training_data_dir2 = get_training_data_dir(DATA_TYPE2, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)

# # For single data

test_dir = '/mnt/ito/test/WD5/test_data/'
data_loader = DataLoaderForTest(test_dir)
test_data, test_water_ids = data_loader.get_test_data_and_water_ids('3smrA')
# For double data
# data_loader = DoubleDataLoader(training_data_dir1, training_data_dir2)
# test_data_displaceable, dis_water_ids = data_loader.get_test_data_and_water_ids(pdb_name, 'displaceable')
# test_data_non_displaceable, non_dis_water_ids = data_loader.get_test_data_and_water_ids(pdb_name, 'non_displaceable')

model_func = ResNet
MODEL_NAME = model_func.__name__
input_shape = (DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, test_data.shape[-1])
n_base = 8
learning_rate = 1e-4
metrics = ['accuracy', dice_coefficient, Recall(), Precision()]
BN = True

# Load the model and its weights
checkpoint_dir = f'../checkpoints/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/aug_train/'
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
model = model_func(n_base, input_shape, learning_rate, BinaryCrossentropy(), metrics, BN=BN)
model.load_weights(latest_checkpoint)

# Predict the labels
prediction = model.predict(test_data)
prediction_values = prediction.reshape(prediction.shape[0])
optimal_threshold_youden = 0.47872123
predicted_labels = custom_threshold(prediction, optimal_threshold_youden)

pos_water_ids = test_water_ids[np.where(predicted_labels == 1)[0]]
neg_water_ids = test_water_ids[np.where(predicted_labels == 0)[0]]
print(pos_water_ids, neg_water_ids)
print(predicted_labels)


# Save and process each type of data
labels = ["pos", "neg"]
ids_list = [pos_water_ids, neg_water_ids]

for label, water_ids in zip(labels, ids_list):
    output_path = f"/mnt/ito/test/WD5/predicted_labeled_water/{holo_name}/{label}_pred_O_placed_{pdb_name}_3.0.pdb"
    input_path = f"/mnt/ito/test/WD5/water_inside_ligand_pocket/{holo_name}/pred_O_placed_{pdb_name}_3.0.pdb"
    
    # Process the predicted data to filter and create a new PDB for each label type
    filter_atoms_and_create_new_pdb(input_path, output_path, water_ids)




