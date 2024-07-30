from trainer.DataGenerator.VoxelDataGenerator import VoxelDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lib.helper import remove_all_checkpoints
import os

def aug_train_func(x_train, y_train, x_val, y_val, input_shape, model_func, epochs, batch_size, learning_rate, n_base, loss, metrics, num_rotations, angle_unit, class_num=1, 
                   BN=False,Sdropout=False, dropout=False, early_stopping=5, model_checkpoint=False, checkpoint_path="",verbose=1, class_weight=None):

    train_generator = VoxelDataGenerator(x_train, y_train, batch_size, num_rotations, angle_unit)
    val_generator = VoxelDataGenerator(x_val, y_val, batch_size, 0, angle_unit)
    # Train the model    
    
    if model_func:
        clf = model_func(n_base=n_base, input_shape=input_shape, loss=loss, metrics=metrics, class_num=class_num, learning_rate=learning_rate ,BN=BN, Sdropout=Sdropout,dropout=dropout)

    callbacks_list = []
    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping, verbose=1, restore_best_weights=True)
        callbacks_list.append(early_stopping)

    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    callbacks_list.append(reduce_lr_callback)

    # to save only the best model

    if model_checkpoint:
        remove_all_checkpoints(os.path.dirname(checkpoint_path))
        model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
        callbacks_list.append(model_checkpoint_callback)
        
    
    if class_weight:
         clf_hist = clf.fit(train_generator, class_weight=class_weight,validation_data=val_generator, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        clf_hist = clf.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    return clf, clf_hist
    
    
    
