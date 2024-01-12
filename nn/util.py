
import pandas as pd
import numpy as np


def sequence_tensor(X, timesteps):
  shifted_series = [X[i:i-timesteps,...] for i in range(timesteps)]
  return np.stack(shifted_series, axis=0).swapaxes(0,1)

def target_array(Y, timesteps):
    arr = Y[timesteps:,...].copy()
    return np.expand_dims(arr, axis=-1)
