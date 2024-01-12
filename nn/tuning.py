
from keras_tuner import Hyperband

from .callbacks import learning_schedule_callback, learning_adjustment_callback, tensorboard_callback

def tune(model, max_epochs=10000, factor=3, directory="./", project_name=None, **fit_kwargs):
    tuner = Hyperband(model, 
                      objective="val_loss", 
                      max_epochs=max_epochs, 
                      factor=factor, 
                      hyperband_iterations=1, 
                      directory=directory, 
                      project_name=project_name)
    fit_kwargs.update(dict(x=model.input, 
                           y=model.input_target, 
                           verbose=2, 
                           callbacks=[learning_schedule_callback, 
                                      learning_adjustment_callback, 
                                      tensorboard_callback], 
                           validation_data=(model.validation, model.validation_target), 
                           shuffle=False))
    tuner.search(**fit_kwargs)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    # best_model.fit(**fit_kwargs)
    return best_model
