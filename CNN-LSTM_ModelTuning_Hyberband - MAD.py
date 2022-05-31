import openpyxl as openpyxl
import pandas as pd
import time
import keras
from data_prep import *
import numpy as np
from itertools import islice
from numpy import array
from numpy import hstack
import tensorflow as tf
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D,MaxPooling1D,Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from keras_tuner.tuners import Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
from kerastuner.tuners import RandomSearch
LOG_DIR = f"{int(time.time())}"

#inputs
n_steps=10
X_train, y_train , X_val , y_val , X_test , y_test , act_label , subject_id_test = load_data_MAD(r'C:\Users\jcb\Desktop\speciale/ChildrenThighSimple10.xlsx',n_steps)
act_label,act_names = shift_act_labels(act_label)

def build_model(hp):
    #Function that defines how to build the model
    input = Input(shape=(10, 5))
    #tunes for hyperparameters in this case number of cells in the LSTM layer
    conv1 = Conv1D(filters=hp.Int('filters_l1',min_value=8,max_value=256,step=8),
                   kernel_size = 3,padding='same', activation='relu')(input)
    conv2 = Conv1D(filters=hp.Int('filters_l2', min_value=8, max_value=256, step=8),
                   kernel_size=3, padding='same', activation='relu')(conv1)
    conv3 = Conv1D(filters=hp.Int('filters_l3', min_value=8, max_value=256, step=8),
                   kernel_size=3, padding='same', activation='relu')(conv2)
    LSTM1 = LSTM(224, return_sequences=True, activation='relu')(conv3)
    LSTM2 = LSTM(64, return_sequences=True, activation='relu')(LSTM1)
    LSTM3 = LSTM(240)(LSTM2)
    output = Dense(1)(LSTM3)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

tuner = Hyperband(
    build_model,
    objective = "val_mean_squared_error",
    max_epochs = 100,
    directory = LOG_DIR
)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=10)
tuner.search(
    x=X_train,
    y=y_train,
    verbose=1,
    epochs=100,
    batch_size=128,
    validation_data=(X_val,y_val),
    callbacks=[stop_early]
)
best_model = tuner.get_best_models()[0]
best_model.summary()
history = best_model.fit(X_train, y_train, epochs=150,validation_data=(X_val,y_val), batch_size=128, verbose=1)
yhat = best_model.predict(X_test, verbose=1)
mse = mean_squared_error(y_test, yhat)
mape = mean_absolute_percentage_error(y_test,yhat)
print('MSE: %.3f MAPE: %.3f%%' % (mse, mape))
print('R^2: %.3f' % r2_score(y_test, yhat))
yhat=np.squeeze(yhat)
print(yhat.shape)
model_name = 'Tuned_Model_CNN-LSTM_MAD'
plot_model(model_name,yhat,y_test,act_label,history,subject_id_test)
save_model(model_name,best_model)
act_boxplot(act_label, act_names,yhat,y_test,model_name)
