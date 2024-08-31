import sys
sys.path.append('../..')

from Class.TestDataLoader import TestDataLoader
from Class.PredictionExecutor import PredictionExecutor
from Class.PredictionAnalyzer import PredictionAnalyzer
from UnkownProteinPredictor import UnkownProteinPredictor
from make_data_folder import copy_holo_files
from lib.path import get_checkpoints_dir

from models.ResNet import ResNet
from lib.helper import get_latest_checkpoint
from custom_losses.dice import dice_coefficient
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf

# GPUを無効化
tf.config.set_visible_devices([], 'GPU')

def predict(protein_name, apo_name, holo_name, model_func, settings, model_params, optimal_threshold):

    output_dir = f"/mnt/ito/test/{protein_name}/predicted_labeled_water/{holo_name}/{settings['CLASSIFYING_RULE']}/"

    model = model_func(**model_params)
    checkpoint_dir = get_checkpoints_dir(**settings)
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    test_dir = f'/mnt/ito/test/{protein_name}/test_data/{apo_name}/'
    test_data_loader = TestDataLoader(test_dir, holo_name)
    prediction_executor = PredictionExecutor(model, latest_checkpoint)
    prediction_analyzer = PredictionAnalyzer(protein_name, apo_name, holo_name, output_dir, optimal_threshold=optimal_threshold)

    unkown_protein_predictor = UnkownProteinPredictor(
        test_data_loader=test_data_loader,
        prediction_executor=prediction_executor,
        prediction_analyzer=prediction_analyzer
    )

    unkown_protein_predictor.run()

    copy_holo_files(protein_name, apo_name, holo_name, settings['CLASSIFYING_RULE'])

if __name__ == '__main__':

    model_func = ResNet
    settings = {
        "DATA_TYPE" : 'gr',
        "DATA_VOXEL_NUM" : 10,
        "CLASSIFYING_RULE" : 'WaterClassifyingRuleSurface',
        "LIGAND_POCKET_DEFINER" : 'LigandPocketDefinerOriginal',
        "LIGAND_VOXEL_NUM" : 8,
        "MODEL_NAME" : model_func.__name__,
        "is_augmented" : True
    }

    model_params = {
        "n_base" : 8,
        "input_shape" : (settings['DATA_VOXEL_NUM']*2+1, settings['DATA_VOXEL_NUM']*2+1, settings['DATA_VOXEL_NUM']*2+1, 1),
        "learning_rate" : 1e-4, 
        "loss" : BinaryCrossentropy(),
        "metrics" : ['accuracy', dice_coefficient, Recall(), Precision()], 
        "BN" : True
    }
    optimal_threshold = 0.45

    protein_names = ['CDK2', 'WD5']
    apo_names = {'CDK2' : ['1hcl', '1pw2'], 'WD5' : ['2h14']}
    holo_names = {'CDK2' : ['1e1v', '1jvp', '1pxi'], 'WD5' : ['3smrA', '3ur4A', '4erqAD']}

    for protein_name in protein_names:
        for apo_name in apo_names[protein_name]:
            for holo_name in holo_names[protein_name]:
                predict(protein_name=protein_name, apo_name=apo_name, holo_name=holo_name, model_func=model_func, settings=settings, model_params=model_params, optimal_threshold=optimal_threshold)
