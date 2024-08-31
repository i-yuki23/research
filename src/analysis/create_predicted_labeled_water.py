import sys
import os
sys.path.append('..')
from lib.path import get_training_data_dir, get_predicted_labeled_water_path, get_displaceable_water_path, get_non_displaceable_water_path
from lib.pdb import filter_atoms_and_create_new_pdb, get_pdb_names_by_txt
from data_loader.SingleDataLoader import SingleDataLoader
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

def extract_ids(labels, ids_array, condition):
    indices = np.where(labels == condition)[0]  # np.where returns indices that match the condition
    return ids_array[indices]

# Define the parameters
DATA_TYPE1 = 'gr'
# DATA_TYPE2 = 'Protein'
DATA_TYPE = f"{DATA_TYPE1}_{DATA_TYPE2}" if 'DATA_TYPE2' in locals() else DATA_TYPE1
is_augmented = True

DATA_VOXEL_NUM = 10
CLASSIFYING_RULE = 'WaterClassifyingRuleEmbedding'
LIGAND_POCKET_DEFINER = 'LigandPocketDefinerOriginal'
LIGAND_VOXEL_NUM = 8


model_func = ResNet
MODEL_NAME = model_func.__name__
input_shape = (DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, 1)
n_base = 8
learning_rate = 1e-4
metrics = ['accuracy', dice_coefficient, Recall(), Precision()]
BN = True
optimal_threshold_youden = 0.45

# Load the model and its weights
checkpoint_dir = f'../checkpoints/valid/{DATA_TYPE}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/aug_train/'
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
model = model_func(n_base, input_shape, learning_rate, BinaryCrossentropy(), metrics, BN=BN)
model.load_weights(latest_checkpoint)



# Load the test data
training_data_dir1 = get_training_data_dir(DATA_TYPE1, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)
# training_data_dir2 = get_training_data_dir(DATA_TYPE2, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)

# # For single data
data_loader = SingleDataLoader(training_data_dir1)

pdb_names = get_pdb_names_by_txt('../../data/valid_test.txt')

for pdb_name in pdb_names:
    test_data_displaceable, dis_water_ids = data_loader.get_test_data_and_water_ids(pdb_name, 'displaceable')
    test_data_non_displaceable, non_dis_water_ids = data_loader.get_test_data_and_water_ids(pdb_name, 'non_displaceable')
    data_dir = '../../data'

    print(optimal_threshold_youden)
    prediction_displaceable = model.predict(test_data_displaceable)
    predicted_labels_dis = custom_threshold(prediction_displaceable, optimal_threshold_youden)

    prediction_non_displaceable = model.predict(test_data_non_displaceable)
    predicted_labels_non_dis = custom_threshold(prediction_non_displaceable, optimal_threshold_youden)

    # Extract IDs based on predictions for all conditions
    water_ids_TP = extract_ids(predicted_labels_dis, dis_water_ids, 1)
    water_ids_FN = extract_ids(predicted_labels_dis, dis_water_ids, 0)
    water_ids_FP = extract_ids(predicted_labels_non_dis, non_dis_water_ids, 1)
    water_ids_TN = extract_ids(predicted_labels_non_dis, non_dis_water_ids, 0)

    water_ids_dis = np.concatenate([water_ids_TP, water_ids_FP])
    water_ids_non = np.concatenate([water_ids_TN, water_ids_FN])

    # Save and process each type of data
    labels = ['displaceable', 'non_displaceable']
    ids_list = [water_ids_dis, water_ids_non]


    output_path_dis = get_predicted_labeled_water_path(DATA_TYPE, pdb_name, DATA_VOXEL_NUM, LIGAND_VOXEL_NUM,
                                                    CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, MODEL_NAME, labels[0], is_augmented)
    input_path_dis = get_displaceable_water_path(pdb_name, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER)

    # Process the predicted data to filter and create a new PDB for each label type
    filter_atoms_and_create_new_pdb(input_path_dis, output_path_dis, ids_list[0])



    output_path_non = get_predicted_labeled_water_path(DATA_TYPE, pdb_name, DATA_VOXEL_NUM, LIGAND_VOXEL_NUM,
                                                    CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, MODEL_NAME, labels[1], is_augmented)
    input_path_non = get_non_displaceable_water_path(pdb_name, LIGAND_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER)

    # Process the predicted data to filter and create a new PDB for each label type
    filter_atoms_and_create_new_pdb(input_path_non, output_path_non, ids_list[1])



