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
def truncate_top_k(x, k=105, inplace=False):
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

# heatmaps = truncate_top_k(heatmaps)
# heatmaps = np.expand_dims(heatmaps, axis=-1)

## Plot the detected regions
top_list = [.05, .1, .2, .3]
onemaps = np.ones_like(heatmaps)
onemaps = np.expand_dims(onemaps, axis=-1)

n_demo, top_demo = 6, 4
fig, axs = plt.subplots(top_demo, n_demo)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.03, hspace=.03)
# random_sample = np.random.randint(n_sample, size=n_demo)
random_sample = [6625, 6097, 4732, 6600, 3287, 973]
for j in range(top_demo):
    heatmap_tmp = truncate_top_k(heatmaps, k=int(top_list[j]*1024))
    heatmap_tmp = np.expand_dims(heatmap_tmp, axis=-1)
    for i in range(n_demo):
        ind_tmp = random_sample[i]
        masked_demo = np.ma.masked_where(heatmap_tmp[ind_tmp] == 0, onemaps[ind_tmp])
        axs[j,i].imshow(x_test[ind_tmp])
        axs[j,i].imshow(masked_demo, 'hsv', alpha=0.5)
        axs[j,i].grid(False)
        axs[j,i].set_xticks([])
        axs[j,i].set_yticks([])
plt.show()

# 6625, 6097, 4732, 6600, 3287, 973