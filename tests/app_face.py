import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
import time
from dnn_inference.DnnT import DnnT

# dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

image_array = np.load('valid_image_array.npy')
label = np.load('valid_label.npy')
image_array /= 255.
image_array = image_array.astype('float32')

oh_label = to_categorical(label)

model_full = tf.keras.applications.resnet.ResNet152(
    include_top = True,
    weights=None, input_shape=(48, 48, 1), classes=7)

model_full.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model_mask = tf.keras.applications.resnet.ResNet152(
    include_top = True,
    weights=None, input_shape=(48, 48, 1), classes=7)

model_mask.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

K.clear_session()

tic = time.perf_counter()

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', mode='max', 
                   verbose=1, patience=15,
                   restore_best_weights=True)

fit_params = {'callbacks': [es],
              'epochs': 500,
              'batch_size': 8,
              'validation_split': .2,
              'verbose': 2}

split_params = {'split': 'one-split',
                'perturb': None,
                'num_perm': 100,
                'ratio_grid': [.2, .4, .6, .8],
                'perturb_grid': [.001, .005, .01, .05, .1],
                'min_inf': 100,
                'min_est': 1000,
                'ratio_method': 'fuse',
                'verbose': 2}

inf_feats = [[np.arange(14,22), np.arange(12,24)],
             [np.arange(14,22), np.arange(30,42)],
             [np.arange(24,32), np.arange(20,29)],
             [np.arange(34,45), np.arange(18,30)]]

shiing = DnnT(inf_feats=inf_feats, 
              model=model_full, model_mask=model_mask, 
              change='mask', eva_metric='zero-one')

p_value_tmp = shiing.testing(image_array, oh_label, 
                            cv_num=5, cp='hommel',
                            fit_params=fit_params,
                            split_params=split_params)

toc = time.perf_counter()

shiing.visual(image_array[100:],oh_label[200:],
              plt_params={'cmap': 'gray', 'alpha':1.})

print('testing time: %.3f' %(toc-tic))
print('P-values: %s' %p_value_tmp)
