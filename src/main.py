import os
import json
from data_loader.SingleDataLoader import SingleDataLoader
from data_loader.DoubleDataLoader import DoubleDataLoader
from models.LeNet import LeNet
from models.AlexNet import Alexnet
from trainer.train import train_func
from lib.path import get_training_data_dir
from custom_losses.dice import dice_loss, dice_coefficient
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy

data_dir = '../data'
train_list = os.path.join(data_dir, 'train_list')
test_list = os.path.join(data_dir, 'test_list')
val_list = os.path.join(data_dir, 'val_list')

DATA_TYPE1 = 'gr'
DATA_TYPE2 = 'gist'
DATA_VOXEL_NUM = 10
CLASSIFYING_RULE = 'WaterClassifyingRuleSurface'
LIGAND_POCKET_DEFINER = 'LigandPocketDefinerOriginal'
LIGAND_VOXEL_NUM = 8

training_data_dir1 = get_training_data_dir(DATA_TYPE1, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)
training_data_dir2 = get_training_data_dir(DATA_TYPE2, DATA_VOXEL_NUM, CLASSIFYING_RULE, LIGAND_POCKET_DEFINER, LIGAND_VOXEL_NUM)

print(training_data_dir1, "\n", training_data_dir2)

data_loader = SingleDataLoader(training_data_dir1)

# data_loader = DoubleDataLoader(training_data_dir1, training_data_dir2)

train_data, train_labels = data_loader.load_data(train_list)
test_data, test_labels = data_loader.load_data(test_list)
val_data, val_labels = data_loader.load_data(val_list)

print('Train data shape: ', train_data.shape)
print('Train labels shape: ', train_labels.shape)
print('Test data shape: ', test_data.shape)
print('Test labels shape: ', test_labels.shape)
print('Val data shape: ', val_data.shape)
print('Val labels shape: ', val_labels.shape)

input_shape = (DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, DATA_VOXEL_NUM*2+1, 1)
epochs = 300
batch_size = 128
n_base = 16
learning_rate = 1e-4
early_stopping = 300
BN = True
dropout = 0.5
model_func = LeNet
losses = [BinaryCrossentropy(), dice_loss]
loss= losses[0]
metrics = ['accuracy', dice_coefficient, Recall(), Precision()]
path_type = '/'.join(training_data_dir1.split('/')[6:11])
checkpoint_path = f"./checkpoints/{path_type}/LeNet/" + "cp-{epoch:04d}.weights.h5"
model_checkpoint = True

# pos = train_labels.sum()
# neg = train_labels.shape[0] - pos
# total = train_labels.shape[0]

# weight_for_0 = (1 / neg) * (total / 2.0)
# weight_for_1 = (1 / pos) * (total / 2.0)
# class_weight = {0: weight_for_0, 1: weight_for_1}

clf, clf_hist, clf_eval = train_func(
                                    x_train=train_data,
                                    y_train=train_labels,
                                    x_test=test_data,
                                    y_test=test_labels,
                                    x_val=val_data,
                                    y_val=val_labels,
                                    input_shape=input_shape,
                                    model_func=model_func,
                                    loss=loss,
                                    metrics=metrics,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    n_base=n_base,
                                    learning_rate=learning_rate,
                                    early_stopping=early_stopping,
                                    checkpoint_path=checkpoint_path,
                                    model_checkpoint=model_checkpoint,
                                    BN = BN,
                                    dropout=dropout
                                )

from lib.helper import make_dir

history_save_path = f"./history/{path_type}/LeNet/training_history.json"
make_dir(history_save_path)
with open(history_save_path, 'w') as f:
    json.dump(clf_hist.history, f)

prediction = clf.predict(test_data)

prediction.round().sum()


