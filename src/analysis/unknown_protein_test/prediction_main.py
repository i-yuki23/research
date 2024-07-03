import sys
sys.path.append('../..')

from Class.TestDataLoader import TestDataLoader
from Class.PredictionExecutor import PredictionExecutor
from Class.PredictionAnalyzer import PredictionAnalyzer
from UnkownProteinPredictor import UnkownProteinPredictor
from lib.path import get_checkpoints_dir

from models.ResNet import ResNet
from lib.helper import get_latest_checkpoint
from custom_losses.dice import dice_coefficient
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import BinaryCrossentropy

def main():
    apo_name = "2h14"
    holo_name = '4erqAD'
    test_dir = f'/mnt/ito/test/{apo_name}/test_data/'
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

    model = model_func(**model_params)
    checkpoint_dir = get_checkpoints_dir(**settings)
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    test_data_loader = TestDataLoader(test_dir, holo_name)
    prediction_executor = PredictionExecutor(model, latest_checkpoint)
    prediction_analyzer = PredictionAnalyzer(apo_name, holo_name, optimal_threshold=0.5)

    unkown_protein_predictor = UnkownProteinPredictor(
        test_data_loader=test_data_loader,
        prediction_executor=prediction_executor,
        prediction_analyzer=prediction_analyzer
    )

    unkown_protein_predictor.run()

if __name__ == '__main__':
    main()
