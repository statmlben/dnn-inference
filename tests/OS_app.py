import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import time
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from DnnT import DnnT

num_classes = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.vstack((x_train, x_test))
y = np.hstack((y_train, y_test))
ind = (y == 9) + (y == 7)
X, y = X[ind], y[ind]
X = X.astype('float32')
X += .01*abs(np.random.randn(14251, 28, 28))
y[y==7], y[y==9] = 0, 1

if K.image_data_format() == 'channels_first':
	X = X.reshape(x.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	X = X.reshape(X.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

X /= 255.

# convert class vectors to binary class matrices
y = keras.utils.to_categorical(y, num_classes)

K.clear_session()

def cnn():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(0.005), metrics=['accuracy'])
	return model

tic = time.perf_counter()
model, model_mask = cnn(), cnn()

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15, restore_best_weights=True)

fit_params = {'callbacks': [es],
			  'epochs': 100,
			  'batch_size': 32,
			  'validation_split': .2,
			  'verbose': 0}

split_params = {'split': 'one-sample',
				'perturb': None,
				'num_perm': 1000,
				'ratio_grid': [.3, .4, .5],
				'perturb_grid': [.005, .01, .05, .1, .5, 1.],
				'min_inf': 100,
				'min_est': 1000,
				'ratio_method': 'close',
				'verbose': 1}

inf_cov = [[np.arange(19,28), np.arange(13,20)], [np.arange(21,28), np.arange(4, 13)],
		   [np.arange(7,16), np.arange(9,16)]]
shiing = DnnT(inf_cov=inf_cov, model=model, model_mask=model_mask, change='mask', eva_metric='zero-one')
p_value_tmp, metric_tmp = shiing.testing(X, y, fit_params=fit_params, split_params=split_params)
toc = time.perf_counter()
print('testing time: %.3f' %(toc-tic))

# OS test: [0.6642883319367069, 0.9672120396024865, 9.977801368581722e-25]
