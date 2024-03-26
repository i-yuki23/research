# %%
import os
from data_load import load_data
from models.LeNet import LeNet
from models.AlexNet import Alexnet
from trainer.train import train_func
from custom_losses.dice import dice_loss, dice_coefficient
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import BinaryFocalCrossentropy

# %%
data_dir = '../data'
train_list = os.path.join(data_dir, 'train_list')
test_list = os.path.join(data_dir, 'test_list')
val_list = os.path.join(data_dir, 'val_list')

# %%
GR_VOXEL_NUM = 10
LIGAND_VOXEL_NUM = 6

# %%
train_data, train_labels = load_data(train_list, LIGAND_VOXEL_NUM, GR_VOXEL_NUM)
test_data, test_labels = load_data(test_list, LIGAND_VOXEL_NUM, GR_VOXEL_NUM)
val_data, val_labels = load_data(val_list, LIGAND_VOXEL_NUM, GR_VOXEL_NUM)

# %%
print('Train data shape: ', train_data.shape)
print('Train labels shape: ', train_labels.shape)
print('Test data shape: ', test_data.shape)
print('Test labels shape: ', test_labels.shape)
print('Val data shape: ', val_data.shape)
print('Val labels shape: ', val_labels.shape)

# %%
input_shape = (GR_VOXEL_NUM*2+1, GR_VOXEL_NUM*2+1, GR_VOXEL_NUM*2+1, 1)
epochs = 300
batch_size = 32
n_base = 32
learning_rate = 1e-5
early_stopping = 300
BN = True
dropout = 0.4
model_func = LeNet
loss= dice_loss
metrics = ['accuracy', dice_coefficient, Recall(), Precision()]
checkpoint_dir = f"./checkpoints/LIGAND_VOXEL_NUM_{LIGAND_VOXEL_NUM}/GR_VOXEL_NUM_{GR_VOXEL_NUM}/LeNet/"
checkpoint_path  = os.path.join(checkpoint_dir, "cp-{epoch:04d}.weights.h5")
model_checkpoint = True

# %%
pos = train_labels.sum()
neg = train_labels.shape[0] - pos
total = train_labels.shape[0]

weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
print(class_weight)

# %%
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
                                    class_weight=class_weight,
                                    BN = BN,
                                    dropout=dropout
                                )

# %%
# prediction = clf.predict(test_data)

# # %% [markdown]
# # 

# # %%
# prediction.round().sum()

# # %%
# precision = 0.756
# recall =0.3592

# 2*precision*recall/(precision+recall)

# %%



