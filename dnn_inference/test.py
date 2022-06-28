import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD

np.random.seed(0)
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

## define the learning models
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
	model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
	return model

model, model_mask = cnn(), cnn()

## fitting param
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=15, restore_best_weights=True)

fit_params = {'callbacks': [es],
			  'epochs': 50,
			  'batch_size': 64,
			  'validation_split': .2,
			  'verbose': 0}

test_params = { 'split': 'one-split',
                'inf_ratio': None,
                'perturb': None,
                'cv_num': 5,
                'cp': 'hommel',
                'verbose': 2}

tune_params = { 'num_perm': 100,
                'ratio_grid': [.2, .4, .6, .8],
                'if_reverse': 0,
                'perturb_range': 2.**np.arange(-5,5),
                'tune_ratio_method': 'fuse',
                'cv_num': 3,
                'cp': 'hommel',
                'verbose': 2}

## Inference based on dnn_inference
from BBoxTest import split_test
## testing based on learning models
inf_feats = [[np.arange(19,28), np.arange(13,20)], [np.arange(21,28), np.arange(4, 13)],[np.arange(7,16), np.arange(9,16)]]
cue = split_test(inf_feats=inf_feats, model_null=model, model_alter=model_mask, eva_metric='zero-one')
cue.tuneHP(k=0, X=X, y=y, test_params=test_params, fit_params=fit_params, tune_params=tune_params, verbose=1)















# cue = split_test(inf_feats=inf_feats, model=model, model_mask=model_mask, change='mask', eva_metric='zero-one')
# p_value_tmp = cue.testing(X, y, cv_num=3, cp='hommel', fit_params=fit_params, split_params=split_params)
# ## visualize testing results
# cue.visual(X,y)
# print('P-values: %s' %p_value_tmp)