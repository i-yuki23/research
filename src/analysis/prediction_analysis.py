# %% [markdown]
# テストデータに対して予測を行う
# →それぞれの水分子の混合行列ごとに分ける
# それぞれの水分子と最近接距離の距離を計算する

# %%
import sys
import os
sys.path.append('..')
from lib.path import get_training_data_dir, get_predicted_labeled_water_path, get_displaceable_water_path, get_non_displaceable_water_path
from lib.pdb import get_atom_coords_by_id, get_coordinates_from_pdb
from data_loader.SingleDataLoader import SingleDataLoader
from data_loader.DoubleDataLoader import DoubleDataLoader
from models.LeNet import LeNet
from models.ResNet import ResNet
from models.ResNet_revised import ResNet_revised
from lib.helper import get_latest_checkpoint
from tensorflow.keras.losses import BinaryCrossentropy
from custom_losses.dice import dice_loss, dice_coefficient
from tensorflow.keras.metrics import Recall, Precision, Accuracy, SpecificityAtSensitivity, AUC, F1Score
from sklearn.metrics import fbeta_score, confusion_matrix
import numpy as np
from lib.plot import plot_histogram
from lib.helper import calculate_closest_distance, calculate_average_of_top_n_closest_distances, calculate_top_n_closest_coords
from SASA.ResSASACalculator import ResSASACalculator

# %%
import tensorflow as tf
# GPUを無効化
tf.config.set_visible_devices([], 'GPU')

# %%
DATA_TYPE1 = 'gr'
DATA_VOXEL_NUM = 20
CLASSIFYING_RULE = 'WaterClassifyingRuleEmbedding'
LIGAND_POCKET_DEFINER = 'LigandPocketDefinerOriginal'
LIGAND_VOXEL_NUM = 8
MODEL_NAME = 'ResNet'
TRAINER_NAME = 'aug_train'
training_data_dir1 = get_training_data_dir(DATA_TYPE1, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)
data_loader = SingleDataLoader(training_data_dir1)

# %%
input_shape = (DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, 1)
n_base = 8
BN = True
checkpoint_dir = f'../checkpoints/valid_all/smoothing/{DATA_TYPE1}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/{TRAINER_NAME}/'
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
model = ResNet(n_base=n_base, input_shape=input_shape, learning_rate=0.0, loss=BinaryCrossentropy(), metrics=[], BN=BN, class_num=2)
model.load_weights(latest_checkpoint)

# %%
def custom_threshold(prediction, threshold):
    return (prediction > threshold).astype(int)
threshold = 0.5

# %%
data_dir = '../../data'
pdb_data_dir = '/mnt/ito/data/pdb_bind'
test_list = os.path.join(data_dir, 'all_valid_test.txt')
with open(test_list, 'r') as f:
    test_pdb_array = f.read().splitlines()
cm_for_each_pdb = {}
accuracy_for_each_pdb = {}
recall_for_each_pdb = {}
precision_for_each_pdb = {}
specificity_for_each_pdb = {}
negative_precision_for_each_pdb = {}

df = {'pdb_id': [], 
      'water_id': [], 
      'prediction_score': [], 
      'prediction_result': [], 
      'nearest_pro_atom_dist': [], 
      'nearest_lig_atom_dist': [], 
      'top_5_avg_pro_atom_dist': [], 
      'top_5_avg_lig_atom_dist': [], 
      'top_10_avg_pro_atom_dist': [],
      'top_10_avg_lig_atom_dist': [],
      'top_20_avg_pro_atom_dist': [],
      'top_20_avg_lig_atom_dist': [],
      'top_30_avg_pro_atom_dist': [],
      'top_30_avg_lig_atom_dist': [],
      'top_40_avg_pro_atom_dist': [],
      'top_40_avg_lig_atom_dist': [],
      'top_50_avg_pro_atom_dist': [],
      'top_50_avg_lig_atom_dist': [],
      'top_5_pro_lig_dist': [],
      'top_10_pro_lig_dist': [],
      'top_20_pro_lig_dist': [],
      'top_30_pro_lig_dist': [],
      'top_40_pro_lig_dist': [],
      'top_50_pro_lig_dist': [],
      'water_sasa': [],
      }
for pdb in test_pdb_array:
    print(pdb)
    try:
        test_data_dis, water_ids_dis = data_loader.get_test_data_and_water_ids(pdb, 'displaceable')
        test_label_dis = np.array([1] * len(water_ids_dis))
        test_data_non_dis, water_ids_non_dis = data_loader.get_test_data_and_water_ids(pdb, 'non_displaceable')
        test_label_non_dis = np.array([0] * len(water_ids_non_dis))

        test_data = np.concatenate([test_data_dis, test_data_non_dis], axis=0)
        test_labels = np.concatenate([test_label_dis, test_label_non_dis], axis=0)
        test_water_ids = np.concatenate([water_ids_dis, water_ids_non_dis], axis=0)

        prediction = model.predict(test_data)
        predicted_label = (prediction[:, 1] >= threshold).astype(int)
        protein_coords = get_coordinates_from_pdb(os.path.join(pdb_data_dir, f'{pdb}/{pdb}_min.pdb'))
        ligand_coords = get_coordinates_from_pdb(os.path.join(pdb_data_dir, f'{pdb}/{pdb}_ligand.pdb'))

        for i in range(len(test_water_ids)):
            # {'pdb_id': '1a4y', 'water_id': 1, 'prediction_score': 0.23, 'prediction_result': 'TP', 'nearest_atom_dist': 2, }
            if test_labels[i] == 1 and predicted_label[i] == 1:
                prediction_result = 'TP'
            elif test_labels[i] == 1 and predicted_label[i] == 0:
                prediction_result = 'FN'
            elif test_labels[i] == 0 and predicted_label[i] == 1:
                prediction_result = 'FP'
            else:
                prediction_result = 'TN'
            df['pdb_id'].append(pdb)
            df['water_id'].append(test_water_ids[i])
            df['prediction_score'].append(prediction[i, 1])
            df['prediction_result'].append(prediction_result)
            water_coords = np.reshape(get_atom_coords_by_id(os.path.join(pdb_data_dir, f'{pdb}/pred_O_placed_{pdb}_3.0.pdb'), test_water_ids[i]), (1, 3))
            df['nearest_pro_atom_dist'].append(calculate_closest_distance(water_coords, protein_coords))
            df['nearest_lig_atom_dist'].append(calculate_closest_distance(water_coords, ligand_coords))
            df['top_5_avg_pro_atom_dist'].append(calculate_average_of_top_n_closest_distances(protein_coords, water_coords, 5))
            df['top_5_avg_lig_atom_dist'].append(calculate_average_of_top_n_closest_distances(ligand_coords, water_coords, 5))
            df['top_10_avg_pro_atom_dist'].append(calculate_average_of_top_n_closest_distances(protein_coords, water_coords, 10))
            df['top_10_avg_lig_atom_dist'].append(calculate_average_of_top_n_closest_distances(ligand_coords, water_coords, 10))
            df['top_20_avg_pro_atom_dist'].append(calculate_average_of_top_n_closest_distances(protein_coords, water_coords, 20))
            df['top_20_avg_lig_atom_dist'].append(calculate_average_of_top_n_closest_distances(ligand_coords, water_coords, 20))
            df['top_30_avg_pro_atom_dist'].append(calculate_average_of_top_n_closest_distances(protein_coords, water_coords, 30))
            df['top_30_avg_lig_atom_dist'].append(calculate_average_of_top_n_closest_distances(ligand_coords, water_coords, 30))
            df['top_40_avg_pro_atom_dist'].append(calculate_average_of_top_n_closest_distances(protein_coords, water_coords, 40))
            df['top_40_avg_lig_atom_dist'].append(calculate_average_of_top_n_closest_distances(ligand_coords, water_coords, 40))
            df['top_50_avg_pro_atom_dist'].append(calculate_average_of_top_n_closest_distances(protein_coords, water_coords, 50))
            df['top_50_avg_lig_atom_dist'].append(calculate_average_of_top_n_closest_distances(ligand_coords, water_coords, 50))
            top_5_pro_coords = calculate_top_n_closest_coords(protein_coords, water_coords, 5)
            top_5_lig_coords = calculate_top_n_closest_coords(ligand_coords, water_coords, 5)
            df['top_5_pro_lig_dist'].append(np.linalg.norm(np.mean(top_5_pro_coords, axis=0) - np.mean(top_5_lig_coords, axis=0)))
            top_10_pro_coords = calculate_top_n_closest_coords(protein_coords, water_coords, 10)
            top_10_lig_coords = calculate_top_n_closest_coords(ligand_coords, water_coords, 10)
            df['top_10_pro_lig_dist'].append(np.linalg.norm(np.mean(top_10_pro_coords, axis=0) - np.mean(top_10_lig_coords, axis=0)))
            top_20_pro_coords = calculate_top_n_closest_coords(protein_coords, water_coords, 20)
            top_20_lig_coords = calculate_top_n_closest_coords(ligand_coords, water_coords, 20)
            df['top_20_pro_lig_dist'].append(np.linalg.norm(np.mean(top_20_pro_coords, axis=0) - np.mean(top_20_lig_coords, axis=0)))
            top_30_pro_coords = calculate_top_n_closest_coords(protein_coords, water_coords, 30)
            top_30_lig_coords = calculate_top_n_closest_coords(ligand_coords, water_coords, 30)
            df['top_30_pro_lig_dist'].append(np.linalg.norm(np.mean(top_30_pro_coords, axis=0) - np.mean(top_30_lig_coords, axis=0)))
            top_40_pro_coords = calculate_top_n_closest_coords(protein_coords, water_coords, 40)
            top_40_lig_coords = calculate_top_n_closest_coords(ligand_coords, water_coords, 40)
            df['top_40_pro_lig_dist'].append(np.linalg.norm(np.mean(top_40_pro_coords, axis=0) - np.mean(top_40_lig_coords, axis=0)))
            top_50_pro_coords = calculate_top_n_closest_coords(protein_coords, water_coords, 50)
            top_50_lig_coords = calculate_top_n_closest_coords(ligand_coords, water_coords, 50)
            df['top_50_pro_lig_dist'].append(np.linalg.norm(np.mean(top_50_pro_coords, axis=0) - np.mean(top_50_lig_coords, axis=0)))
            res_sasa_calculator = ResSASACalculator(os.path.join(pdb_data_dir, f'{pdb}/water_sasa/water_sasa_{test_water_ids[i]}.pdb'), 'BCD')
            df['water_sasa'].append(res_sasa_calculator.calculate_res_sasa())
    except Exception as e:
        continue        



# %%


# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
pd.DataFrame(df).to_csv('prediction_analysis.csv', index=False)

#