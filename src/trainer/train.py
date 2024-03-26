from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from custom_losses.dice import dice_coefficient



def train_func(x_train, y_train, x_val, y_val, x_test, y_test, input_shape, model_func, epochs, batch_size, learning_rate, n_base, loss, metrics, class_num=1, 
               BN=False,Sdropout=False, dropout=False, early_stopping=5, model_checkpoint=False, checkpoint_path="",verbose=1, class_weight=None):

    if model_func:
        clf = model_func(n_base=n_base, input_shape=input_shape, loss=loss, metrics=metrics, class_num=class_num, learning_rate=learning_rate ,BN=BN, Sdropout=Sdropout,dropout=dropout)

    callbacks_list = []
    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping, verbose=1, restore_best_weights=True)
        callbacks_list.append(early_stopping)

    # to save only the best model

    if model_checkpoint:
        model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor=dice_coefficient,
        mode='max',
        save_best_only=True)
        callbacks_list.append(model_checkpoint_callback)
        
    
    if class_weight:
         clf_hist = clf.fit(x_train, y_train, class_weight=class_weight,validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,callbacks=callbacks_list, verbose=verbose)
    else:
        clf_hist = clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,callbacks=callbacks_list, verbose=verbose)
    print("start to evaluate the model")
    clf_eval=clf.evaluate(x_test,y_test,verbose=0)

    return clf, clf_hist, clf_eval 