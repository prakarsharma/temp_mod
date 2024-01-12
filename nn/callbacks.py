
from datetime import datetime

from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard


def early_stopping(min_delta=0, patience=1, start_from_epoch=1):
    return EarlyStopping(monitor='val_loss', 
                         min_delta=min_delta, 
                         patience=patience, 
                         start_from_epoch=start_from_epoch)
early_stopping_callback = early_stopping(min_delta=1e-4, 
                                         patience=10, 
                                         start_from_epoch=100)

def learning_schedule(decay=0.995, min_lr=1e-6, start_from_epoch=1):
    def lr_decay(epoch, lr):
        return lr if epoch < start_from_epoch else max(lr * decay, min_lr)
    return LearningRateScheduler(lr_decay)
learning_schedule_callback = learning_schedule(decay=0.995, 
                                               min_lr=1e-6, 
                                               start_from_epoch=100)

def learning_adjustment(factor=0.5, patience=1, min_lr=1e-6):
    return ReduceLROnPlateau(monitor='val_loss', 
                             factor=factor, 
                             patience=patience, 
                             min_lr=min_lr)
learning_adjustment_callback = learning_adjustment(factor=0.5, 
                                                   patience=10, 
                                                   min_lr=1e-6)

def tensorboard(log_dir="/Users/prakarsharma/Documents/Projects/DF accelerator/analyses/logs/"):
    log_dir = log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
    return TensorBoard(log_dir=log_dir, 
                       histogram_freq=1, 
                       write_graph=True, 
                       write_images=True, 
                       write_steps_per_second=True, 
                       update_freq="epoch")
tensorboard_callback = tensorboard()
