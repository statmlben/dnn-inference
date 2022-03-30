import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
import time
from tensorflow_addons.optimizers import AdamW, SGDW
from BBoxTest import ada_split_test
from tensorflow.keras.models import Sequential
from alexnet import load_dataset, preprocess_image, build_model
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50

np.random.seed(7)
K.clear_session()

(x_train, y_train), (x_test, y_test) = load_dataset()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
# x_test = np.array([preprocess_image(x_test[i], 224, 224) for i in range(len(x_test))])  # (224, 224, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

n_sample, height, width, n_channel = x_test.shape

# heatmaps = np.zeros_like(x_test[:,:,:,0], dtype="float32")
heatmaps = np.load('heatmaps.npy')
# for i in range(n_sample):
#     heatmaps[i] = cv2.resize(heatmap[i], (height, width))  # resize to image size

## generate mask data
def truncate_top_k(x, k=50, inplace=False):
    arr_shape = x.shape
    m = arr_shape[0]
    x = x.reshape(m,-1)
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = x[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = x < kth_vals[:, None]
    # replace mask by 0
    if not inplace:
        x = np.where(is_smaller_than_kth, 0, x)
        return x.reshape(arr_shape)
    x[is_smaller_than_kth] = 0
    return x

heatmaps = truncate_top_k(heatmaps)
heatmaps = np.expand_dims(heatmaps, axis=-1)

x_perm = np.random.permutation(x_test)
z_test = x_test.copy()
z_test = np.where(heatmaps == 0., x_test, x_perm)

# ## Plot the detected regions
# n_demo = 5
# fig, axs = plt.subplots(2, n_demo)
# for i in range(n_demo):
#     ind_tmp = np.random.randint(n_sample)
#     masked_demo = np.ma.masked_where(heatmaps[ind_tmp] == 0, heatmaps[ind_tmp])
#     axs[0,i].imshow(x_test[ind_tmp])
#     axs[0,i].imshow(masked_demo, 'Reds', interpolation='none', alpha=0.7)
#     axs[1,i].imshow(z_test[ind_tmp])
# plt.show()

## define model
def ResNet50_model(input_shape, num_classes=100):
    resnet_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
    model = Sequential()
    # model.add(UpSampling2D())
    # model.add(UpSampling2D())
    # model.add(UpSampling2D())
    model.add(resnet_model)
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dense(2048, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(512, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=SGDW(learning_rate=.0001, weight_decay=.0000, momentum=.9),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_full = ResNet50_model(input_shape=(height, width, n_channel))
model_mask = ResNet50_model(input_shape=(height, width, n_channel))

# model_full = ResNet50(input_shape=(height, width, n_channel), classes=100, weights=None)
# model_mask = ResNet50(input_shape=(height, width, n_channel), classes=100, weights=None)

# model_full.compile(optimizer=SGDW(learning_rate=.01, weight_decay=.000, momentum=.9),
#               loss='categorical_crossentropy', metrics=['accuracy'])
# model_mask.compile(optimizer=SGDW(learning_rate=.01, weight_decay=.000, momentum=.9),
#               loss='categorical_crossentropy', metrics=['accuracy'])

## fitting parameters
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

scheduler = ReduceLROnPlateau(monitor='loss', mode='min',
                            factor=0.382, patience=3, verbose=True, min_lr=1e-8)

es = EarlyStopping(monitor='val_loss', mode='min',
                   verbose=1, patience=30,
                   restore_best_weights=True)

tic = time.perf_counter()

fit_params = {'callbacks': [scheduler, es],
              'epochs': 300,
              'batch_size': 32,
              'validation_split': .2,
              'verbose': 2}

split_params = {'split': 'two-split',
                'num_perm': 100,
                'ratio_grid': [.2, .4, .6, .8],
                'perturb_scale': 5,
                'min_inf': 100,
                'min_est': 100,
                'ratio_method': 'log',
                'verbose': 0}

## test the performance for the model
# model_full.fit(x_test, y_test, **fit_params)

shiing = ada_split_test(model=model_full, model_mask=model_mask,
                        change='mask', loss='CategoricalCrossentropy')

p_value_tmp = shiing.testing(x_test, z_test, y_test, perturb=-5,
                            cv_num=5, cp='hommel',
                            fit_params=fit_params,
                            split_params=split_params)

print('P-values: %s' %p_value_tmp)

## Top 10
# p_value = 1; 
# [[0.9886371997279508,
#   0.49098338838733097,
#   0.8185027875313721,
#   0.8649043609978169,
#   0.36558705250490675]]

## Top 30
# P-values: 0.2739683671733749
# [[0.9535828528189415,
#   0.5119071528406658,
#   0.9683620414483622,
#   0.6535360287238807,
#   0.023997229241463497]]
