
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
# from keras.optimizers import Adam

from keras_tuner import HyperModel

from .util import sequence_tensor, target_array
from .metrics import CRM
from .callbacks import learning_schedule_callback, learning_adjustment_callback, tensorboard_callback


class LSTMnet(HyperModel):
    def __init__(self, 
                 timesteps:int, 
                 horizon:int, 
                 train:pd.DataFrame, 
                 target:pd.DataFrame, 
                 test:pd.DataFrame=None, 
                 truth:pd.DataFrame=None):
        self.timesteps = timesteps
        self.horizon = horizon        
        self.input = sequence_tensor(train.values, timesteps)
        self.input_target = target_array(target.values, timesteps)
        self.validation = sequence_tensor(test.values, timesteps)
        self.validation_target = target_array(truth.values, timesteps)

    def make_model(self, recurrent_units, dropout):
        model = Sequential(name="LSTM_dropout")
        model.add(Input(shape=self.input.shape[1:])) # input shape is time_steps x features
        lstm = LSTM(units=recurrent_units,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    dropout=0,
                    recurrent_dropout=0,
                    return_sequences=False,
                    return_state=False,
                    stateful=False,
                    unroll=False)
        model.add(lstm)
        model.add(Dropout(dropout))
        model.add(Dense(units=self.horizon, 
                        activation="relu"))
        # model.build()
        self.model = model

    def summary(self):
        if hasattr(self, "model"):
            print(self.model.summary())

    def weights(self):
        if hasattr(self, "model"):
            for L in self.model.layers:
                for _ in L.weights:
                    print(_.name, "->", _.shape, sep=" ")

    def compile_model(self, lr):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        self.model.compile(loss="mse", 
                           optimizer=optimizer, 
                           metrics=[CRM])

    def fit_model(self, **fit_kwargs):
        fit_kwargs.update(dict(x=self.input, 
                            y=self.input_target, 
                            verbose=2, 
                            callbacks=[learning_schedule_callback, 
                                        learning_adjustment_callback, 
                                        tensorboard_callback], 
                            validation_data=(self.validation, self.validation_target), 
                            shuffle=False))
        self.history = self.model.fit(**fit_kwargs)

    def build(self, hp):
        recurrent_units = hp.Int(name="recurrent_units", 
                                 min_value=32, 
                                 max_value=512, 
                                 step=32, 
                                 sampling="linear")
        dropout = hp.Float(name="dropout", 
                           min_value=0.1, 
                           max_value=0.4, 
                           step=0.1, 
                           sampling="linear")
        lr = hp.Float(name="lr", 
                      min_value=1e-6, 
                      max_value=0.01, 
                      step=10, 
                      sampling="log")
        self.make_model(recurrent_units, dropout)
        self.compile_model(lr)
        return self.model

    def plot_training(self):
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def predict(self, X):
        return self.model.predict(X)[:,0]
