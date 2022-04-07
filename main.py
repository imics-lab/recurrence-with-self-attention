#!/usr/bin/python3
"""
Authors: Alexander Katrompas, Theodoros Ntakouris, Vangelis Metsis
Organization: Texas State University

Driver code from the paper titled "Recurrence and Self-Attention vs the
Transformer for Time-Series Classification: A Comparative Study."

Usage: main.py [1|2] [-vg]
       (assuming python3 in /usr/bin/)

1: LSTM (default)
2: Transformer

v: verbose mode (optional)
g: graphing mode (optional)

"""

# python libs
from sys import version
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from matplotlib import pyplot
from numpy import argmax

# local libs
import cfg #importing local library cfg
import functions as fn
from selfAttn import SelfAttn

print("TensorFlow Version:", tf. __version__)
print("Python Version:", version)

# get command line arguments
model, verbose, graph = fn.get_args()

#load data
train_X, train_Y, test_X, test_Y, valid_X, valid_Y = fn.load_data(cfg.TRAIN, cfg.TEST, index=cfg.INDEX, header=cfg.HEADER)

if verbose:
    print("Shapes loaded...")
    print("  train X", train_X.shape)
    print("  train Y", train_Y.shape)
    print("  test_X", test_X.shape)
    print("  test_Y", test_Y.shape)
    print("  valid_X", valid_X.shape)
    print("  valid_Y", valid_Y.shape)
    print()

train_X = fn.shape_3D_data(train_X, cfg.SEQLENGTH)
test_X  = fn.shape_3D_data(test_X,  cfg.SEQLENGTH)
valid_X = fn.shape_3D_data(valid_X, cfg.SEQLENGTH)
train_Y = fn.shape_3D_data(train_Y, cfg.SEQLENGTH)
test_Y  = fn.shape_3D_data(test_Y,  cfg.SEQLENGTH)
valid_Y = fn.shape_3D_data(valid_Y, cfg.SEQLENGTH)

if verbose:
    print("3D Shaped...")
    print("  train X", train_X.shape)
    print("  train Y", train_Y.shape)
    print("  test_X", test_X.shape)
    print("  test_Y", test_Y.shape)
    print("  valid_X", valid_X.shape)
    print("  valid_Y", valid_Y.shape)
    print()

if model == 1:
    if verbose:
        if not cfg.LSTM_ATTENTION:
            print("LSTM")
            print("=====================")
        else:
            print("LSTM w/SELF-ATTENTION")
            print("=====================")

    model = tf.keras.models.Sequential()
    model.add(layers.Input(shape=(cfg.SEQLENGTH, train_X.shape[2])))
    model.add(layers.LSTM(cfg.LSTM, input_shape=(cfg.SEQLENGTH,
              train_X.shape[2]), 
              return_sequences=True,
              dropout=cfg.DROPOUT))
    
    if cfg.LSTM_ATTENTION:
        model.add(SelfAttn(cfg.SEQLENGTH))
        model.add(layers.Dense(cfg.DENSE1, activation='sigmoid'))
        model.add(layers.Dense(cfg.DENSE2, activation='sigmoid'))
    else:
        model.add(layers.TimeDistributed(layers.Dense(cfg.DENSE1, activation='sigmoid')))
        model.add(layers.Dense(cfg.DENSE2, activation='sigmoid'))
    model.add(layers.Dropout(cfg.DROPOUT))

    model.add(layers.Dense(cfg.OUTPUT, activation='sigmoid' if cfg.OUTPUT == 1 else 'softmax'))

elif model == 2:
    if verbose:
        print("TRANSFORMER")
        print("===========")

    inputs = tf.keras.Input(shape=(cfg.SEQLENGTH, train_X.shape[2]))
    x = inputs

    for _ in range(cfg.NUM_ATTN_LAYERS):
        x = fn.transformer_encoder(x, cfg.HEAD_SIZE, cfg.NUM_HEADS, cfg.FF_DIM, cfg.DROPOUT)

    for dim in cfg.MLP_DIMS:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.Dropout(cfg.DROPOUT)(x)

    outputs = layers.Dense(cfg.OUTPUT, activation='sigmoid' if cfg.OUTPUT == 1 else 'softmax')(x)
    model = tf.keras.Model(inputs, outputs)

#############################
# Finalize Model and Train
#############################
if cfg.OUTPUT == 1:
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'])
    #model.compile(loss='mse', metrics=['accuracy'])
else:
    model.compile(loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
    
if verbose:
    model.summary()
    print("Epochs:", cfg.EPOCHS)

early_stopping = EarlyStopping(patience=cfg.PATIENCE, restore_best_weights=True, verbose=verbose)
history = model.fit(train_X, train_Y,
                #batch_size = cfg.BATCH_SIZE,
                epochs=cfg.EPOCHS,
                verbose=verbose,
                shuffle=cfg.SHUFFLE,
                validation_data=(test_X, test_Y),
                callbacks=[early_stopping])

#############################
# Graphing Loss
#############################
if len(history.history['loss']) and len(history.history['val_loss']) and graph:
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

#############################
# Prediction and Validation
#############################
y = valid_Y.flatten()
yhat = model.predict(valid_X)

if cfg.OUTPUT == 1:
    yhat = yhat.flatten()
else:
    ytemp = yhat.reshape(yhat.shape[0]*yhat.shape[1], yhat.shape[2])
    yhat = []
    for value in ytemp:
        yhat.append(argmax(value))
    del ytemp

# save predictions
f = open("valid_out.csv", "w")
count = 0
for value in yhat:
    f.write(str(y[count]) + "," + str(value) + "\n")
    count += 1
f.close()