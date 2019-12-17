import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


checkpoint_path = "training_1/cp.ckpt"

lin_reg = keras.Sequential([
  layers.Dense(64, activation='relu', input_shape=[1]),
  layers.Dense(64, activation='relu', ),
  layers.Dense(1)
])

lin_reg.load_weights(checkpoint_path)

print('Example Result: ')
example_result = lin_reg.predict( np.array([80.0,] ) )
print(example_result)
