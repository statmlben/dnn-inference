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
def truncate_top_k(x, k=300, inplace=False):
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

split_params = {'split': 'one-split',
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

p_value_tmp = shiing.testing(x_test, z_test, y_test, perturb=0.,
                            cv_num=5, cp='hommel',
                            fit_params=fit_params,
                            split_params=split_params)

print('P-values: %s' %p_value_tmp)


## Top 30
# P-values: 0.2739683671733749
# [[0.9535828528189415,
#   0.5119071528406658,
#   0.9683620414483622,
#   0.6535360287238807,
#   0.023997229241463497]]

## Top 50
# P-values: 0.5806301844542555
# [[0.9042093765401764,
#   0.6017056883707457,
#   0.5203411062344576,
#   0.05085811834635815,
#   0.7574058713907246]]

## Top 100
## P-values: 0.4585041725704668
# [[0.6660837337572938,
#   0.07067262780073709,
#   0.640995322559567,
#   0.08032189884446134,
#   0.2668277389693443]]

## Top 105
# P-values: 0.1585338369012804

# In [2]: shiing.p_values_comb
# Out[2]: 
# [[0.9346554018382566,
#   0.04145731200359863,
#   0.013886175495002663,
#   0.2925353927939932,
#   0.2582654355939047]]


## Top 150
# P-values: 0.022025477697054225
# cv: 0; p_value: 0.23730; metric: 4.18029(1.64289); metric_mask: 4.23737(1.38276)
# cv: 1; p_value: 0.00193; metric: 4.08803(1.60177); metric_mask: 4.33238(1.48058)
# cv: 2; p_value: 0.07902; metric: 4.14023(1.64442); metric_mask: 4.26915(1.60445)
# cv: 3; p_value: 0.14095; metric: 4.11274(1.74915); metric_mask: 4.20935(1.49170)
# cv: 4; p_value: 0.08525; metric: 4.24369(1.58590); metric_mask: 4.36743(1.50846)

## Top 200
## P-values: 0.020581694881040676
# cv: 0; p_value: 0.10505; metric: 4.21296(1.70401); metric_mask: 4.31898(1.42195)
# cv: 1; p_value: 0.00180; metric: 4.14183(1.60516); metric_mask: 4.37981(1.35567)
# cv: 2; p_value: 0.16104; metric: 4.28272(1.65719); metric_mask: 4.36463(1.29607)
# cv: 3; p_value: 0.09234; metric: 4.15739(1.71338); metric_mask: 4.26920(1.33686)
# cv: 4; p_value: 0.01071; metric: 4.22072(1.77530); metric_mask: 4.42197(1.21797)

## Top 300
# P-values: 9.680534110929853e-05
# [[0.0002591352132558976,
#   0.0002876840244149727,
#   0.0008009028034353621,
#   0.0013067155863826427,
#   8.479299951179434e-06]]

## one-split Test
## Top 50
# ##################################################
# Accept H0 with p_value: 0.056
# ##################################################
# P-values: 0.05590673174603811

# In [2]: shiing.p_values_comb
# Out[2]: 
# [[0.3126856181914892,
#   0.2818213233686754,
#   0.4769886117875321,
#   0.004896940006952243,
#   0.010392334097586591]]


## Top 105
##################################################
# P-values: 0.004080783199873713

# In [2]: shiing.p_values_comb
# Out[2]: 
# [[0.45322920846240417,
#   0.00035744086422251505,
#   0.13921609291040987,
#   0.0064364270222994355,
#   0.003933718244770822]]

## Top 150
# cv: 0; p_value: 0.04744; metric: 4.18963(1.69364); metric_mask: 4.31006(1.40904)
# cv: 1; p_value: 0.20878; metric: 4.24925(1.76978); metric_mask: 4.30419(1.55505)
# cv: 2; p_value: 0.00533; metric: 4.10944(1.56418); metric_mask: 4.31769(1.45557)
# cv: 3; p_value: 0.00000; metric: 4.04358(1.57913); metric_mask: 4.24466(1.41945)
# cv: 4; p_value: 0.00397; metric: 4.18272(1.78149); metric_mask: 4.34764(1.54833)

# Reject H0 with p_value: 0.000
# ##################################################
# P-values: 1.9237501904652553e-05

# Top 200
# Reject H0 with p_value: 0.000
# ##################################################
# P-values: 0.0001347751306288199

# In [2]: shiing.p_values_comb
# Out[2]: 
# [[0.00016717501434148529,
#   0.00010396586420072425,
#   0.00034843246135700097,
#   0.00012447313028519555,
#   1.1805120930991523e-05]]

# Top 300
# ##################################################
# Reject H0 with p_value: 0.000
# ##################################################
# P-values: 1.1217476783345067e-07

# In [2]: shiing.p_values_comb
# Out[2]: 
# [[4.254120228194746e-08,
#   1.9651054218998657e-08,
#   0.00013608696610330437,
#   1.0696503039978701e-08,
#   3.049070115043304e-05]]
