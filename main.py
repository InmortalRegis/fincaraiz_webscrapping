import locale
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

import numpy as np


dftrain = pd.read_csv('aptos_train.csv')
dfeval = pd.read_csv('aptos_eval.csv', header=None)
dfeval.columns = ["areas","precios"]

x_train = dftrain.pop('areas').values
y_train = dftrain.pop('precios').values
x_test = dfeval.pop('areas').values
y_test =  dfeval.pop('precios').values

X = tf.constant(x_train)
Y = tf.constant(y_train)
X_test = tf.constant(x_test)
Y_test = tf.constant(y_test)



lin_reg = keras.Sequential([
  layers.Dense(64, activation='relu', input_shape=[1]),
  layers.Dense(64, activation='relu', ),
  layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)
lin_reg.compile(
  loss='mse',
  optimizer=optimizer,
  metrics=['mse', 'mae']
)



checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


history = lin_reg.fit(
  x=X,
  y=Y,
  shuffle=True,
  epochs=1000,
  validation_split=0.2,
  verbose=1,
  callbacks=[cp_callback]
)


print('Evaluation:')
evaluate = lin_reg.evaluate(X_test, Y_test, verbose=2)

# class LinearModel:
#     def __call__(self, x):
#         return self.W * x + self.b

#     def __init__(self):
#         self.W = tf.Variable(tf.zeros([1]))
#         self.b = tf.Variable(tf.ones([1]))


# model = LinearModel()
# plt.scatter(X, Y, label="true")
# plt.scatter(X, model(X), label="predicted")
# plt.legend()


# def loss(Y, y_pred):
#     return tf.reduce_mean(tf.square(Y - y_pred))

# def train(linear_model, X, Y, lr=0.001):
#     with tf.GradientTape() as t:
#         current_loss = loss(Y, linear_model(X))

#     dW, db = t.gradient(current_loss, [linear_model.W, linear_model.b])
#     linear_model.W.assign_sub(lr * dW)
#     linear_model.b.assign_sub(lr * db)


# model = LinearModel()
# Ws, bs = [], []
# epochs = 100
# for epoch in range(epochs):
#     Ws.append(model.W.numpy())
#     bs.append(model.b.numpy())

#     real_loss = loss(Y, model(X))

#     train(model, X, Y, lr=0.0001)
#     print(f"Epoch {epoch}: Ws: {Ws[-1]} Bs: {bs[-1]} Loss: {real_loss.numpy()}")

# # plt.plot(range(epochs), Ws, 'r', range(epochs), bs, 'b')
# # plt.legend(['W', 'b'])
# plt.show()