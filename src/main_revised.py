import os
import json
from data_loader.SingleDataLoader import SingleDataLoader
from models.ResNet_revised import ResNet_revised
from trainer.train import train_func
from trainer.aug_train import aug_train_func
from lib.path import get_training_data_dir
from custom_losses.dice import dice_loss, dice_coefficient
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import BinaryCrossentropy

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


data_dir = '../data'
train_list = os.path.join(data_dir, 'train.txt')
test_list = os.path.join(data_dir, 'test.txt')
val_list = os.path.join(data_dir, 'val.txt')

DATA_TYPE1 = 'gr'
DATA_VOXEL_NUM = 10
CLASSIFYING_RULE = 'WaterClassifyingRuleEmbedding'
LIGAND_POCKET_DEFINER = 'LigandPocketDefinerOriginal'
LIGAND_VOXEL_NUM = 8

training_data_dir1 = get_training_data_dir(DATA_TYPE1, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)

data_loader = SingleDataLoader(training_data_dir1)

train_data, train_labels = data_loader.load_data(train_list)
test_data, test_labels = data_loader.load_data(test_list)
val_data, val_labels = data_loader.load_data(val_list)

print('Train data shape: ', train_data.shape)
print('Train labels shape: ', train_labels.shape)
print('Test data shape: ', test_data.shape)
print('Test labels shape: ', test_labels.shape)
print('Val data shape: ', val_data.shape)
print('Val labels shape: ', val_labels.shape)

input_shape = (DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, train_data.shape[-1])
epochs = 300
batch_size = 128
n_base = 16
learning_rate = 1e-4
early_stopping = 30
BN = True
dropout = 0.5
model_func = ResNet_revised
MODEL_NAME = model_func.__name__
TRAINER_NAME = 'aug_train'
losses = [BinaryCrossentropy(), dice_loss]
loss= losses[0]
metrics = ['accuracy', dice_coefficient, Recall(), Precision()]
path_type = f'/{DATA_TYPE1}/data_voxel_num_{DATA_VOXEL_NUM}/{LIGAND_POCKET_DEFINER}/ligand_pocket_voxel_num_{LIGAND_VOXEL_NUM}/{CLASSIFYING_RULE}/{MODEL_NAME}/{TRAINER_NAME}/'

checkpoint_path = f"./checkpoints/{path_type}/" + "cp-{epoch:04d}.weights.h5"
model_checkpoint = True

pos = train_labels.sum()
neg = train_labels.shape[0] - pos
total = train_labels.shape[0]

weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
print(class_weight)

clf, clf_hist = aug_train_func(
                                x_train=train_data,
                                y_train=train_labels,
                                x_val=val_data,
                                y_val=val_labels,
                                input_shape=input_shape,
                                model_func=model_func,
                                loss=loss,
                                metrics=metrics,
                                epochs=epochs,
                                batch_size=batch_size,
                                num_rotations=3,
                                angle_unit=45,
                                n_base=n_base,
                                learning_rate=learning_rate,
                                early_stopping=early_stopping,
                                checkpoint_path=checkpoint_path,
                                model_checkpoint=model_checkpoint,
                                class_weight=class_weight,
                                BN = BN,
                                dropout=dropout
                            )

# clf, clf_hist, clf_eval = train_func(
#                                     x_train=train_data,
#                                     y_train=train_labels,
#                                     x_test=test_data,
#                                     y_test=test_labels,
#                                     x_val=val_data,
#                                     y_val=val_labels,
#                                     input_shape=input_shape,
#                                     model_func=model_func,
#                                     loss=loss,
#                                     metrics=metrics,
#                                     epochs=epochs,
#                                     batch_size=batch_size,
#                                     n_base=n_base,
#                                     learning_rate=learning_rate,
#                                     early_stopping=early_stopping,
#                                     checkpoint_path=checkpoint_path,
#                                     model_checkpoint=model_checkpoint,
#                                     class_weight=class_weight,
#                                     BN = BN,
#                                     dropout=dropout
#                                 )

from lib.helper import make_dir

history_save_path = f"./history/{path_type}/training_history.json"
make_dir(history_save_path)
with open(history_save_path, 'w') as f:
    json.dump(clf_hist.history, f)

prediction = clf.predict(test_data)

prediction.round().sum()

