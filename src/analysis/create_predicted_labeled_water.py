import sys
import os
sys.path.append('..')
from lib.path import get_training_data_dir, get_predicted_labeled_water_path, get_displaceable_water_path, get_non_displaceable_water_path
from lib.pdb import filter_atoms_and_create_new_pdb
from data_loader.DataLoader import DataLoader
from models.LeNet import LeNet
from lib.helper import get_latest_checkpoint
from tensorflow.keras.losses import BinaryCrossentropy
from custom_losses.dice import dice_loss, dice_coefficient
from tensorflow.keras.metrics import Recall, Precision
import numpy as np

# Define the parameters
pdb_name = '4lkk'
DATA_TYPE = 'gr'
DATA_VOXEL_NUM = 10
CLASSIFYING_RULE = 'WaterClassifyingRuleSurface'
LIGAND_POCKET_DEFINER = 'LigandPocketDefinerOriginal'
LIGAND_VOXEL_NUM = 8
MODEL_NAME = 'LeNet'
input_shape = (DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, 1)
n_base = 16
learning_rate = 1e-5
metrics = ['accuracy', dice_coefficient, Recall(), Precision()]
BN = True

# Load the test data
training_data_dir = get_training_data_dir(DATA_TYPE, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)
data_loader = DataLoader(training_data_dir)
test_data_displaceable, dis_water_ids = data_loader.get_test_data_and_water_ids(pdb_name, os.path.join(training_data_dir, 'displaceable/'))
test_data_non_displaceable, non_dis_water_ids = data_loader.get_test_data_and_water_ids(pdb_name, os.path.join(training_data_dir, 'non_displaceable/'))

# Load the model and its weights
checkpoint_dir = f'../checkpoints/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/'
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
model = LeNet(n_base, input_shape, learning_rate, BinaryCrossentropy(), metrics, BN=BN)
model.load_weights(latest_checkpoint)

# Predict the labels
prediction_displaceable = model.predict(test_data_displaceable)
predicted_labels_dis = prediction_displaceable.round()

prediction_non_displaceable = model.predict(test_data_non_displaceable)
predicted_labels_non_dis = prediction_non_displaceable.round()

def extract_ids(labels, ids_array, condition):
    indices = np.where(labels == condition)[0]  # np.where returns indices that match the condition
    return ids_array[indices]

# Extract IDs based on predictions for all conditions
water_ids_TP = extract_ids(predicted_labels_dis, dis_water_ids, 1)
water_ids_FN = extract_ids(predicted_labels_dis, dis_water_ids, 0)
water_ids_FP = extract_ids(predicted_labels_non_dis, non_dis_water_ids, 1)
water_ids_TN = extract_ids(predicted_labels_non_dis, non_dis_water_ids, 0)

# Save and process each type of data
labels = ["TP", "FN", "FP", "TN"]
ids_list = [water_ids_TP, water_ids_FN, water_ids_FP, water_ids_TN]
print(ids_list)
for label, water_ids in zip(labels[:2], ids_list[:2]):
    output_path = get_predicted_labeled_water_path(DATA_TYPE, pdb_name, DATA_VOXEL_NUM, LIGAND_VOXEL_NUM,
                                                   CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, MODEL_NAME, label)
    input_path = get_displaceable_water_path(pdb_name, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER)
    
    # Process the predicted data to filter and create a new PDB for each label type
    filter_atoms_and_create_new_pdb(input_path, output_path, water_ids)

for label, water_ids in zip(labels[2:], ids_list[2:]):
    output_path = get_predicted_labeled_water_path(DATA_TYPE, pdb_name, DATA_VOXEL_NUM, LIGAND_VOXEL_NUM,
                                                   CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, MODEL_NAME, label)
    input_path = get_non_displaceable_water_path(pdb_name, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER)
    
    # Process the predicted data to filter and create a new PDB for each label type
    filter_atoms_and_create_new_pdb(input_path, output_path, water_ids)



